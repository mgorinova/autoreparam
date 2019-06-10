from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import numpy as np

import util
import interleaved

from tensorflow_probability.python import mcmc
from tensorflow.python.ops.parallel_for import pfor

from tensorflow.python.framework import smart_cond

FLAGS = tf.app.flags.FLAGS


def find_best_learning_rate(elbo,
                            variational_parameters,
                            learnable_parameters_prior=None,
                            learnable_parameters=None):
  """
                Optimises the given ELBO using different learning rates.
                Returns the best initial step-size for HMC, together with
                information regarding the best optimisation find.
                If `learnable_parameters` is given, it also returns the
                best parameterisation for the model.
        """
  best_timeline = []
  best_elbo_with_prior = None
  best_prior_logp = None
  best_lr = None

  step_size_approx = util.get_approximate_step_size(
      variational_parameters, num_leapfrog_steps=1)  #FLAGS.num_leapfrog_steps)

  learning_rate_ph = tf.placeholder(shape=[], dtype=tf.float32)
  learning_rate = tf.Variable(learning_rate_ph, trainable=False)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # If specified, incorporate a prior on the learnable parameters.
  prior_logp = tf.constant(0., dtype=elbo.dtype)
  if learnable_parameters_prior is not None:
    prior_logp = sum([tf.reduce_sum(learnable_parameters_prior.log_prob(param))
                      for param in learnable_parameters.values()])
  elbo_with_prior = elbo + prior_logp

  train = optimizer.minimize(-elbo_with_prior)
  init = tf.global_variables_initializer()

  def get_learning_rate(step, base_learning_rate):
    if step > 2* FLAGS.num_optimization_steps / 3:
      return base_learning_rate / 20
    elif step > FLAGS.num_optimization_steps / 3:
      return base_learning_rate / 5
    else:
      return base_learning_rate

  for learning_rate_val in FLAGS.learning_rates:
    with tf.Session() as sess:

      feed_dict = {learning_rate_ph: learning_rate_val}
      sess.run(init, feed_dict=feed_dict)

      elbo_with_prior_timeline = []
      prior_logp_timeline = []

      for step in range(FLAGS.num_optimization_steps):
        _, e, plp = sess.run([train, elbo_with_prior, prior_logp],
                             feed_dict={learning_rate: get_learning_rate(
                                 step, learning_rate_val)})
        elbo_with_prior_timeline.append(e)
        prior_logp_timeline.append(plp)

      this_elbo_with_prior = np.mean(elbo_with_prior_timeline[-32:])
      this_prior_logp = np.mean(prior_logp_timeline[-32:])
      info_str = ('     finished optimization with elbo {} vs '
                  'best ELBO {}'.format(this_elbo_with_prior,
                                        best_elbo_with_prior))
      util.print(info_str)
      if not np.isfinite(this_elbo_with_prior): continue

      if (best_elbo_with_prior is None
          or best_elbo_with_prior < this_elbo_with_prior):

        best_elbo_with_prior = this_elbo_with_prior
        best_prior_logp = this_prior_logp
        best_timeline = elbo_with_prior_timeline
        best_lr = learning_rate_val

        step_size_init = sess.run(step_size_approx)

        vals = sess.run(list(variational_parameters.values()))
        learned_variational_params = collections.OrderedDict(
            zip(variational_parameters.keys(), vals))

        if learnable_parameters is not None:
          vals = sess.run(list(learnable_parameters.values()))
          learned_reparam = collections.OrderedDict(
              zip(learnable_parameters.keys(), vals))
        else:
          learned_reparam = None

  # Return a 'pure' ELBO for valid comparisons with other methods.
  best_elbo = best_elbo_with_prior - best_prior_logp

  return (best_elbo, best_timeline, best_lr, step_size_init,
          learned_variational_params, learned_reparam)


# Dave's code:
def vectorized_sample(model, model_args, num_samples):
  """Draw multiple joint samples from an Ed2 model."""

  def loop_body(i):  # trace the model to draw a single joint sample
    with ed.tape() as model_tape:
      model(*model_args)
    # pfor works with Tensors only, so extract RV values
    values = collections.OrderedDict(
        (k, rv.value) for k, rv in model_tape.items())
    return values

  return pfor(loop_body, num_samples)


def vectorize_log_joint_fn(log_joint_fn):
  """Convert a function of Tensor args into a vectorized fn."""

  # @tfe.function(autograph=False)
  def vectorized_log_joint_fn(*args, **kwargs):
    x1 = args[0] if len(args) > 0 else kwargs.values()[0]

    num_inputs = x1.shape[0]
    if not x1.shape.is_fully_defined():
      num_inputs = tf.shape(x1)[0]

    def loop_body(i):
      sliced_args = [tf.gather(v, i) for v in args]
      sliced_kwargs = {k: tf.gather(v, i) for k, v in kwargs.items()}
      return log_joint_fn(*sliced_args, **sliced_kwargs)

    result = pfor(loop_body, num_inputs)
    result.set_shape([num_inputs])
    return result

  return vectorized_log_joint_fn


def hmc(target, model, model_config, step_size_init, initial_states, reparam):
  """Runs HMC to sample from the given target distribution."""
  if reparam == 'CP':
    to_centered = lambda x: x
  elif reparam == 'NCP':
    to_centered = model_config.to_centered
  else:
    to_centered = model_config.make_to_centered(**reparam)

  model_config = model_config._replace(to_centered=to_centered)

  initial_states = list(initial_states)  # Variational samples.
  shapes = [s[0].shape for s in initial_states]

  vectorized_target = vectorize_log_joint_fn(target)

  per_chain_initial_step_sizes = [
      np.array(step_size_init[i] * np.ones(initial_states[i].shape) /
               (FLAGS.num_leapfrog_steps / 4.)**2).astype(np.float32)
      for i in range(len(step_size_init))
  ]

  kernel = mcmc.SimpleStepSizeAdaptation(
      inner_kernel=mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=vectorized_target,
          step_size=per_chain_initial_step_sizes,
          num_leapfrog_steps=FLAGS.num_leapfrog_steps),
      adaptation_rate=0.05,
      num_adaptation_steps=FLAGS.num_adaptation_steps)

  states_orig, kernel_results = mcmc.sample_chain(
      num_results=FLAGS.num_samples,
      num_burnin_steps=FLAGS.num_burnin_steps,
      current_state=initial_states,
      kernel=kernel,
      num_steps_between_results=1)

  states_transformed = transform_mcmc_states(states_orig, to_centered)
  ess = tfp.mcmc.effective_sample_size(states_transformed)

  return states_orig, kernel_results, states_transformed, ess


def vectorise_transform(transform):

  def vtransf(many_chains_sample):

    def loop_body(c):
      return transform(
          [tf.gather(rv_states, c) for rv_states in many_chains_sample])

    return pfor(loop_body, FLAGS.num_chains)

  return vtransf


def hmc_interleaved(model_config, target_cp, target_ncp, num_leapfrog_steps_cp,
                    num_leapfrog_steps_ncp, step_size_cp, step_size_ncp,
                    initial_states_cp):

  model_cp = model_config.model

  initial_states = list(initial_states_cp)  # Variational samples.
  shapes = [s[0].shape for s in initial_states]

  cp_step_sizes = [
      np.array(
          np.ones(
              shape=np.concatenate([[FLAGS.num_chains], shapes[i]]).astype(int))
          * step_size_cp[i],
          dtype=np.float32) / np.float32((num_leapfrog_steps_cp / 4.)**2)
      for i in range(len(step_size_cp))
  ]

  ncp_step_sizes = [
      np.array(
          np.ones(
              shape=np.concatenate([[FLAGS.num_chains], shapes[i]]).astype(int))
          * step_size_ncp[i],
          dtype=np.float32) / np.float32((num_leapfrog_steps_ncp / 4.)**2)
      for i in range(len(step_size_ncp))
  ]

  vectorized_target_cp = vectorize_log_joint_fn(target_cp)
  vectorized_target_ncp = vectorize_log_joint_fn(target_ncp)

  inner_kernel_cp = mcmc.SimpleStepSizeAdaptation(
      inner_kernel=mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=vectorized_target_cp,
          step_size=cp_step_sizes,
          num_leapfrog_steps=num_leapfrog_steps_cp),
      adaptation_rate=0.05,
      target_accept_prob=0.75,
      num_adaptation_steps=FLAGS.num_adaptation_steps)

  inner_kernel_ncp = mcmc.SimpleStepSizeAdaptation(
      inner_kernel=mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=vectorized_target_ncp,
          step_size=ncp_step_sizes,
          num_leapfrog_steps=num_leapfrog_steps_ncp),
      adaptation_rate=0.05,
      target_accept_prob=0.75,
      num_adaptation_steps=FLAGS.num_adaptation_steps)

  to_centered = model_config.to_centered
  to_noncentered = model_config.to_noncentered

  kernel = interleaved.Interleaved(inner_kernel_cp, inner_kernel_ncp,
                                   vectorise_transform(to_centered),
                                   vectorise_transform(to_noncentered))

  states, kernel_results = mcmc.sample_chain(
      num_results=FLAGS.num_samples,
      num_burnin_steps=FLAGS.num_burnin_steps,
      current_state=initial_states,
      kernel=kernel,
      num_steps_between_results=1)

  ess = tfp.mcmc.effective_sample_size(states)

  return states, kernel_results, ess


def transform_mcmc_states(states, transform_fn):
  """Transforms all states using the provided transform function."""

  num_samples = FLAGS.num_samples
  num_chains = FLAGS.num_chains

  def loop_body(sample_idx):

    def loop_body_chain(chain_idx):
      print('\nNested pfor!\n')
      return transform_fn([
          tf.gather(tf.gather(rv_states, sample_idx), chain_idx)
          for rv_states in states
      ])

    return pfor(loop_body_chain, num_chains)

  return pfor(loop_body, num_samples)
