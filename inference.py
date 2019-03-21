from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import numpy as np
import google3.experimental.users.davmre.autoreparam.util as util
import google3.experimental.users.davmre.autoreparam.interleaved as interleaved

from tensorflow_probability.python import mcmc
from google3.third_party.tensorflow.python.ops.parallel_for import pfor

from google3.third_party.tensorflow.python.framework import smart_cond

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

  for learning_rate_val in FLAGS.learning_rates:
    with tf.Session() as sess:

      feed_dict = {learning_rate_ph: learning_rate_val}
      sess.run(init, feed_dict=feed_dict)

      elbo_with_prior_timeline = []
      prior_logp_timeline = []
      for _ in range(FLAGS.num_optimization_steps):
        _, e, plp = sess.run([train, elbo_with_prior, prior_logp])
        elbo_with_prior_timeline.append(e)
        prior_logp_timeline.append(plp)

      this_elbo_with_prior = np.mean(elbo_with_prior_timeline[-32:])
      this_prior_logp = np.mean(prior_logp_timeline[-32:])
      info_str = ('     finished optimization with elbo {} vs '
                  'best ELBO {}'.format(this_elbo_with_prior,
                                        best_elbo_with_prior))
      util.print(info_str)
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


# Copied from tfp.mcmc.util
def is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))


def make_per_chain_step_size_update_policy(num_adaptation_steps,
                                           target_rate=0.75,
                                           decrement_multiplier=0.01,
                                           increment_multiplier=0.01,
                                           step_counter=None):
  if step_counter is None and num_adaptation_steps is not None:
    step_counter = tf.compat.v1.get_variable(
        name='step_size_adaptation_step_counter',
        initializer=np.array(-1, dtype=np.int64),
        # Specify the dtype for variable sharing to work correctly
        # (b/120599991).
        dtype=tf.int64,
        trainable=False,
        use_resource=True)

  def step_size_simple_update_fn(step_size_var, kernel_results):
    if kernel_results is None:
      if is_list_like(step_size_var):
        return [tf.identity(ss) for ss in step_size_var]
      return tf.identity(step_size_var)

    decrement_locs = kernel_results.log_accept_ratio < tf.cast(
        tf.math.log(target_rate), kernel_results.log_accept_ratio.dtype)
    broadcast_ones = tf.ones_like(kernel_results.log_accept_ratio)
    adjustment = tf.where(
        decrement_locs,
        -decrement_multiplier / (1. + decrement_multiplier) * broadcast_ones,
        increment_multiplier * broadcast_ones)

    def assign_step_size_var(step_size_var):
      adjustment_rank, step_size_rank = tf.rank(adjustment), tf.rank(
          step_size_var)
      # `expand_dims` breaks on zero-length `dims`, so avoid calling it if we
      # don't actually need to add any broadcasting dims.
      broadcasted_adjustment = smart_cond.smart_cond(
          adjustment_rank < step_size_rank, lambda: tf.expand_dims(
              adjustment, tf.range(adjustment_rank - step_size_rank, 0)),
          lambda: adjustment)
      return step_size_var.assign_add(
          step_size_var * tf.cast(broadcasted_adjustment, step_size_var.dtype))

    """
                def assign_step_size_var(step_size_var):
                        needed_dims = tf.range(tf.rank(adjustment) -
                        tf.rank(step_size_var), 0)

                        expanded = tf.expand_dims(adjustment, needed_dims) if
                        needed_dims.shape != (0,) \
                                else adjustment

                        broadcasted_adjustment = tf.cast(expanded,
                        step_size_var.dtype)
                        return step_size_var.assign_add(step_size_var *
                        broadcasted_adjustment)
                """

    def build_assign_op():
      if is_list_like(step_size_var):
        return [assign_step_size_var(ss) for ss in step_size_var]
      return assign_step_size_var(step_size_var)

    if num_adaptation_steps is None:
      return build_assign_op()
    else:
      with tf.control_dependencies([step_counter.assign_add(1)]):
        return tf.cond(
            pred=step_counter < num_adaptation_steps,
            true_fn=build_assign_op,
            false_fn=lambda: step_size_var)

  return step_size_simple_update_fn


def hmc(target, model, model_config, step_size_init, reparam):
  """Runs HMC to sample from the given target distribution."""
  if reparam == 'CP':
    to_centered = lambda x: x
  elif reparam == 'NCP':
    to_centered = model_config.to_centered
  else:
    to_centered = model_config.make_to_centered(**reparam)

  model_config = model_config._replace(to_centered=to_centered)

  initial_states = [value for (param, value) in \
    vectorized_sample(model, model_config.model_args,
    num_samples=FLAGS.num_chains).items() if \
    param not in model_config.observed_data.keys()]

  initial_states = list(initial_states)

  shapes = [s[0].shape for s in initial_states]

  vectorized_target = vectorize_log_joint_fn(target)

  #FIXME: possibly initialise at a variational sample rather than the prior?

  v_step_size = [
      tf.get_variable(
          name='step_size' + str(i),
          initializer=np.array(np.ones(
                shape=np.concatenate([[FLAGS.num_chains],
                                      shapes[i]]).astype(int)) *
              step_size_init[i],
              dtype=np.float32) / np.float32(
                  (FLAGS.num_leapfrog_steps / 4.)**2),
          use_resource=True,  # For TFE compatibility.
          trainable=False) for i in range(len(step_size_init))
  ]

  kernel = mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=vectorized_target,
      step_size=v_step_size,
      num_leapfrog_steps=FLAGS.num_leapfrog_steps,
      step_size_update_fn=make_per_chain_step_size_update_policy(
          num_adaptation_steps=FLAGS.num_adaptation_steps,
            target_rate=0.75,
            increment_multiplier=0.05,
            decrement_multiplier=0.05))

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


def hmc_interleaved(model_config,
                    target_cp,
                    target_ncp,
                    step_size_cp,
                    step_size_ncp):

  model_cp = model_config.model

  initial_states = [value for (param, value) in \
           vectorized_sample(model_cp, model_config.model_args,
           num_samples=FLAGS.num_chains).items() if \
           param not in model_config.observed_data.keys()]

  initial_states = list(initial_states)

  shapes = [s[0].shape for s in initial_states]

  cp_step_sizes = [
      tf.get_variable(
          name='step_size_cp' + str(i),
          initializer=np.array(np.ones(
                shape=np.concatenate([[FLAGS.num_chains],
                                      shapes[i]]).astype(int)) *
              step_size_cp[i],
              dtype=np.float32) / np.float32(
                  (FLAGS.num_leapfrog_steps / 4.)**2),
          use_resource=True,  # For TFE compatibility.
          trainable=False) for i in range(len(step_size_cp))
  ]

  ncp_step_sizes = [
      tf.get_variable(
          name='step_size_ncp' + str(i),
          initializer=np.array(np.ones(
                shape=np.concatenate([[FLAGS.num_chains],
                                      shapes[i]]).astype(int)) *
              step_size_ncp[i],
              dtype=np.float32) / np.float32(
                  (FLAGS.num_leapfrog_steps / 4.)**2),
          use_resource=True,  # For TFE compatibility.
          trainable=False) for i in range(len(step_size_ncp))
  ]

  vectorized_target_cp = vectorize_log_joint_fn(target_cp)
  vectorized_target_ncp = vectorize_log_joint_fn(target_ncp)

  step_counter_cp = tf.compat.v1.get_variable(
      name='step_size_adaptation_step_counter_cp',
      initializer=np.array(-1, dtype=np.int64),
      # Specify the dtype for variable sharing to work correctly
      # (b/120599991).
      dtype=tf.int64,
      trainable=False,
      use_resource=True)

  step_counter_ncp = tf.compat.v1.get_variable(
      name='step_size_adaptation_step_counter_ncp',
      initializer=np.array(-1, dtype=np.int64),
      # Specify the dtype for variable sharing to work correctly
      # (b/120599991).
      dtype=tf.int64,
      trainable=False,
      use_resource=True)

  inner_kernel_cp = mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=vectorized_target_cp,
      step_size=cp_step_sizes,
      num_leapfrog_steps=FLAGS.num_leapfrog_steps,
      step_size_update_fn=make_per_chain_step_size_update_policy(
          num_adaptation_steps=FLAGS.num_adaptation_steps,
            target_rate=0.75,
            increment_multiplier=0.05,
            decrement_multiplier=0.05,
            step_counter=step_counter_cp))

  inner_kernel_ncp = mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=vectorized_target_ncp,
      step_size=ncp_step_sizes,
      num_leapfrog_steps=FLAGS.num_leapfrog_steps,
      step_size_update_fn=make_per_chain_step_size_update_policy(
          num_adaptation_steps=FLAGS.num_adaptation_steps,
            target_rate=0.75,
            increment_multiplier=0.05,
            decrement_multiplier=0.05,
            step_counter=step_counter_ncp))

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
