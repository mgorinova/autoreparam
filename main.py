# python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import collections
from collections import OrderedDict

import io

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import google3.experimental.users.davmre.autoreparam.inference as inference
import google3.experimental.users.davmre.autoreparam.graphs as graphs
import google3.experimental.users.davmre.autoreparam.models as models
import google3.experimental.users.davmre.autoreparam.util as util

import google3.experimental.users.davmre.autoreparam.program_transformations as ed_transforms

from tensorflow_probability.python import mcmc
from google3.third_party.tensorflow.python.ops.parallel_for import pfor

import json

flags = tf.app.flags

flags.DEFINE_string('model', default='8schools', help='Model to be used.')

flags.DEFINE_string('dataset', default='', help='Dataset to be used.')

flags.DEFINE_string(
    'inference', default='VI', help='Inference method to be used: VI or HMC.')

flags.DEFINE_string(
    'method',
    default='CP',
    help='Method to be used: CP, NCP, i (only if inference = HMC), cVIP, dVIP.')

flags.DEFINE_string(
    'learnable_parmeterisation_type',
    default='exp',
    help='Type of learnable parameterisation. Either `exp` or `scale`.')

flags.DEFINE_boolean(
    'reparameterise_variational',
    default=False,
    help='Whether or not to reparameterise the variational model too.')

flags.DEFINE_boolean(
    'discrete_prior',
    default=False,
    help='Whether to use a prior encouraging the parameterisation parameters'
          'to be 0 or 1.')


flags.DEFINE_string('results_dir', default='', help='File to write results.')

flags.DEFINE_list(
    'learning_rates',
    default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
    help='Learning rates (list)')

flags.DEFINE_integer(
    'num_optimization_steps',
    default=3000,
    help='Number of steps to optimize the ELBO.')

flags.DEFINE_integer(
    'num_mc_samples',
    default=256,
    help='Number of Monte Carlo samples to use in the ELBO.')

flags.DEFINE_integer(
    'num_leapfrog_steps', default=None, help='Number of leapfrog steps.')

flags.DEFINE_list(
    'num_leapfrog_steps_list',
    default=[1, 2, 4, 8, 16, 32, 64, 128],
    help='Number of leapfrog steps options.')

flags.DEFINE_integer(
    'num_samples', default=50000, help='Number of HMC samples.')

flags.DEFINE_integer(
    'num_tuning_samples', default=10000, help='Number of HMC samples to tune the chains.')

flags.DEFINE_integer('num_chains', default=100, help='Number of HMC chains.')

flags.DEFINE_integer(
    'num_burnin_steps', default=10000, help='Number of warm-up steps.')

flags.DEFINE_integer(
    'num_adaptation_steps', default=6000, help='Number of adaptation steps.')

flags.DEFINE_string('f', default='', help='kernel')

FLAGS = flags.FLAGS


def create_target_graph(model_config, results_dir):

  cVIP_path = os.path.join(
    results_dir, 'cVIP_{}{}{}.json'.format(
        FLAGS.learnable_parmeterisation_type,
        '_reparam_variational' if FLAGS.reparameterise_variational else '',
        '_discrete_prior' if FLAGS.discrete_prior else ''))

  actual_reparam = None
  if FLAGS.method == 'CP':
    (target, model, elbo, variational_parameters,
     learnable_parameters) = graphs.make_cp_graph(model_config)
    actual_reparam = 'CP'
  elif FLAGS.method == 'NCP':
    (target, model, elbo, variational_parameters,
     learnable_parameters) = graphs.make_ncp_graph(model_config)
    actual_reparam = 'NCP'
  elif FLAGS.method == 'i':
    if FLAGS.inference == 'VI':
      Exception('Cannot run interleaved VI. Use `i` method with HMC only.')
    target_cp, model_cp, _, _, _ = graphs.make_cp_graph(model_config)
    target_ncp, model_ncp, _, _, _ = graphs.make_ncp_graph(model_config)
    target = (target_cp, target_ncp)
    model = (model_cp, model_ncp)
    elbo, variational_parameters, learnable_parameters = None, None, None

  elif FLAGS.method == 'cVIP':
    if FLAGS.inference == "VI":
      (target, model, elbo, variational_parameters,
       learnable_parameters) = graphs.make_cvip_graph(
           model_config,
           parameterisation_type=FLAGS.learnable_parmeterisation_type)
    else:
      with tf.gfile.Open(cVIP_path, 'r') as f:
        prev_results = json.load(f)
        actual_reparam = prev_results['learned_reparam']

        (target, model, elbo, variational_parameters,
         learnable_parameters) = graphs.make_dvip_graph(
             model_config,
             actual_reparam,
             parameterisation_type=FLAGS.learnable_parmeterisation_type)

  elif FLAGS.method == 'dVIP':
    if tf.gfile.Exists(cVIP_path):
      with tf.gfile.Open(cVIP_path, 'r') as f:
        prev_results = json.load(f)
        reparam = prev_results['learned_reparam']
    else:
      raise Exception('Run cVIP first to find reparameterisation')

    discrete_parameterisation = collections.OrderedDict(
        [(key, (np.array(reparam[key]) >= 0.5).astype(np.float32))
         for key in reparam.keys()])
    print("discrete parameterisation is", discrete_parameterisation)

    (target, model, elbo, variational_parameters,
     learnable_parameters) = graphs.make_dvip_graph(
         model_config,
         discrete_parameterisation,
         parameterisation_type=FLAGS.learnable_parmeterisation_type)
    actual_reparam = discrete_parameterisation

  return (target,
          model,
          elbo,
          variational_parameters,
          learnable_parameters,
          actual_reparam)


def tune_leapfrog_steps(model_config,
                        num_tuning_samples,
                        initial_step_size,
                        results_dir):
  util.print('Tuning number of leapfrog steps...')

  num_samples_original = FLAGS.num_samples
  FLAGS.num_samples = num_tuning_samples
  max_nls = 1
  max_nls_value = 0

  for nls in FLAGS.num_leapfrog_steps_list:
    util.print('\nTrying out {} leapfrog steps\n'.format(nls))
    FLAGS.num_leapfrog_steps = nls

    (target, model, elbo, variational_parameters,
     learnable_parameters, actual_reparam) = create_target_graph(
            model_config, results_dir)

    if FLAGS.method == 'i':
      (states_orig, kernel_results, ess) = \
        inference.hmc_interleaved(
         model_config,
         *target,
         step_size_cp=initial_step_size[0],
         step_size_ncp=initial_step_size[1])

    else:
      (states_orig, kernel_results, states, ess) = \
        inference.hmc(target, model, model_config,
                      initial_step_size,
                      reparam=actual_reparam)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

      init.run()
      ess_final = sess.run(ess)

    #ess_final = [e / FLAGS.num_leapfrog_steps for e in ess_final]
    ess_min, sem_min = util.get_min_ess(ess_final)
    ess_min = ess_min / FLAGS.num_leapfrog_steps
    sem_min = sem_min / FLAGS.num_leapfrog_steps

    if ess_min > max_nls_value:
      max_nls_value = ess_min
      max_nls = nls

  FLAGS.num_samples = num_samples_original

  return max_nls


def main(_):

  #tf.logging.set_verbosity(tf.logging.ERROR)
  np.warnings.filterwarnings('ignore')

  util.print('Loading model {} with dataset {}.'.format(FLAGS.model,
                                                        FLAGS.dataset))

  if FLAGS.model == 'radon':
    model_config = models.get_radon(state_code=FLAGS.dataset)
  elif FLAGS.model == 'radon_stddvs':
    model_config = models.get_radon_model_stddvs(state_code=FLAGS.dataset)
  elif FLAGS.model == '8schools':
    model_config = models.get_eight_schools()
  elif FLAGS.model == 'german_credit_gammascale':
    model_config = models.get_german_credit_gammascale()
    util.print('Warning! This model contains Gamma variables, which ' +
               'causes problems with `parallel_for`.')
  elif FLAGS.model == 'german_credit_lognormalcentered':
    model_config = models.get_german_credit_lognormalcentered()
  elif FLAGS.model == 'election':
    model_config = models.get_election()
  elif FLAGS.model == 'electric':
    model_config = models.get_electric()
  elif FLAGS.model == 'time_series':
    model_config = models.get_time_series()
  elif FLAGS.model == 'neals_funnel':
    model_config = models.get_neals_funnel()
  else:
    raise Exception('unknown model {}'.format(FLAGS.model))

  if FLAGS.results_dir == '':
    results_dir = FLAGS.model + '_' + FLAGS.dataset
  else:
    results_dir = FLAGS.results_dir

  if not tf.gfile.Exists(results_dir):
    tf.gfile.MakeDirs(results_dir)

  filename = '{}{}{}{}.json'.format(
      FLAGS.method,
      ('_' +
       FLAGS.learnable_parmeterisation_type if 'VIP' in FLAGS.method else ''),
      ('_reparam_variational'
       if 'VIP' in FLAGS.method and FLAGS.reparameterise_variational else ''),
      ('_discrete_prior'
       if 'VIP' in FLAGS.method and FLAGS.discrete_prior else ''))

  file_path = os.path.join(results_dir, filename)

  if FLAGS.inference == 'VI':

    (target, model, elbo, variational_parameters,
     learnable_parameters, actual_reparam) = create_target_graph(
            model_config, results_dir)

    if tf.gfile.Exists(file_path):
      util.print(
          'Already ran experiment {}-{} on model {} with dataset {}. Skipping'
          .format(FLAGS.inference, FLAGS.method, FLAGS.model, FLAGS.dataset))
      return

    learnable_parameters_prior = None
    if FLAGS.discrete_prior:
      # Use a mixture of Laplace (as opposed to Beta or Kumaraswamy) because
      # it takes finite values at 0 and 1.
      learnable_parameters_prior = tfp.distributions.Mixture(
          tfp.distributions.Categorical(logits=[0., 5., 0.]),
          [tfp.distributions.Laplace(loc=0., scale=0.1),
           tfp.distributions.Uniform(),
           tfp.distributions.Laplace(loc=1., scale=0.1)])

    (elbo_final, elbo_timeline, learning_rate, initial_step_size,
     learned_variational_params,
     learned_reparam) = inference.find_best_learning_rate(
         elbo,
         variational_parameters,
         learnable_parameters_prior=learnable_parameters_prior,
         learnable_parameters=learnable_parameters)

    # Save actual parameters used for dVIP
    if learned_reparam is None:
      learned_reparam = actual_reparam

    results = {
     'elbo': elbo_final.item(),
     'estimated_elbo_std': (np.std(elbo_timeline[-32:])).item(),
     'learning_rate': learning_rate,
     'initial_step_size': [i.item() if np.isscalar(i) else i.tolist() \
                for i in initial_step_size],
     'learned_reparam': OrderedDict([
       (k, learned_reparam[k].item() \
        if np.isscalar(learned_reparam[k]) \
        else learned_reparam[k].tolist()) \
       for k in learned_reparam.keys() ]) \
      if learned_reparam is not None else None,
    }

    with tf.gfile.Open(file_path, 'w') as outfile:
      json.dump(results, outfile)

  elif FLAGS.inference == 'HMC':

    if FLAGS.method == 'i':

      filename_cp = 'CP.json'
      filename_ncp = 'NCP.json'

      file_path_cp = os.path.join(results_dir, filename_cp)
      file_path_ncp = os.path.join(results_dir, filename_ncp)

      if tf.gfile.Exists(file_path_cp) and tf.gfile.Exists(file_path_ncp):
        with tf.gfile.Open(file_path_cp, 'r') as f:
          prev_results = json.load(f)
          initial_step_size_cp = prev_results['initial_step_size']

        with tf.gfile.Open(file_path_ncp, 'r') as f:
          prev_results = json.load(f)
          initial_step_size_ncp = prev_results['initial_step_size']

        try:
          with tf.gfile.Open(file_path, 'r') as f:
            num_ls = prev_results.get('num_leapfrog_steps', None)
        except:
          num_ls = None
      else:
        raise Exception('Run VI first to find initial step sizes')

      if num_ls is None and FLAGS.num_leapfrog_steps is None:

        max_nls = tune_leapfrog_steps(
            model_config,
            FLAGS.num_tuning_samples,
            (initial_step_size_cp, initial_step_size_ncp),
            results_dir)

        with tf.gfile.Open(file_path, 'w') as f:
          prev_results['num_leapfrog_steps'] = max_nls
          json.dump(prev_results, f)

        FLAGS.num_leapfrog_steps = max_nls

      elif FLAGS.num_leapfrog_steps is None:
        FLAGS.num_leapfrog_steps = num_ls

      util.print('\nNumber of leaprog steps is set to {}.\n'.format(
          FLAGS.num_leapfrog_steps))

      (target, model, elbo, variational_parameters,
       learnable_parameters, actual_reparam) = create_target_graph(
              model_config, results_dir)

      target_cp, target_ncp = target


      (states, kernel_results, ess) = \
       inference.hmc_interleaved(model_config, target_cp, target_ncp,
                            step_size_cp=initial_step_size_cp,
                step_size_ncp=initial_step_size_ncp)

      init = tf.global_variables_initializer()
      with tf.Session() as sess:

        init.run()

        start_time = time.time()

        samples, is_accepted, ess_final = sess.run(
            (states, kernel_results.is_accepted, ess))

        mcmc_time = time.time() - start_time

    else:
      if tf.gfile.Exists(file_path):
        with tf.gfile.Open(file_path, 'r') as f:
          prev_results = json.load(f)
          initial_step_size = prev_results['initial_step_size']
          num_ls = prev_results.get('num_leapfrog_steps', None)
      else:
        raise Exception('Run VI first to find initial step sizes')

      if num_ls is None and FLAGS.num_leapfrog_steps is None:

        max_nls = tune_leapfrog_steps(model_config,
                                      FLAGS.num_tuning_samples,
                                      initial_step_size,
                                      results_dir)

        with tf.gfile.Open(file_path, 'w') as f:
          prev_results['num_leapfrog_steps'] = max_nls
          json.dump(prev_results, f)

        FLAGS.num_leapfrog_steps = max_nls

      elif FLAGS.num_leapfrog_steps is None:
        FLAGS.num_leapfrog_steps = num_ls

      util.print('\nNumber of leaprog steps is set to {}.\n'.format(
          FLAGS.num_leapfrog_steps))

      (target, model, elbo, variational_parameters,
       learnable_parameters, actual_reparam) = create_target_graph(
           model_config, results_dir)

      (states_orig, kernel_results, states, ess) = \
        inference.hmc(target, model, model_config, initial_step_size,
                      reparam=(actual_reparam
                               if actual_reparam is not None
                               else learned_reparam))

      init = tf.global_variables_initializer()
      with tf.Session() as sess:

        init.run()

        start_time = time.time()

        samples, is_accepted, ess_final, samples_orig = sess.run(
            (states, kernel_results.is_accepted, ess,
             states_orig))

        mcmc_time = time.time() - start_time

    ess_min, sem_min = util.get_min_ess(ess_final)
    ess_min = ess_min / FLAGS.num_leapfrog_steps
    sem_min = sem_min / FLAGS.num_leapfrog_steps

    results = prev_results

    def init_results(list):
      for l in list:
        if l not in results.keys():
          results[l] = []

    init_results([
        'ess_min',
        'sem_min',
        'acceptance_rate',
        'mcmc_time_sec',
        #'step_size',
    ])

    results.get('ess_min', []).append(ess_min.item())
    results.get('sem_min', []).append(sem_min.item())

    util.print('ESS: {} +/- {}'.format(ess_min, sem_min))

    results.get('acceptance_rate', []).append(
        (np.sum(is_accepted) * 100. /
         float(FLAGS.num_samples * FLAGS.num_chains)).item())

    results.get('mcmc_time_sec', []).append(mcmc_time)

    with tf.gfile.Open(file_path, 'w') as outfile:
      json.dump(results, outfile)

    i = len(results['ess_min'])

    with ed.tape() as model_tape:
            model_config.model(*model_config.model_args)
    param_names = [k for k in list(model_tape.keys()) if k not in model_config.observed_data]
    print(param_names)
    print(len(samples))

    dict_res = dict([(param_names[i], samples[i]) for i in range(len(param_names))])
    dict_ess = dict([(param_names[i], np.array(ess_final[i])) for i in range(len(param_names))])

    # Work around issues saving np arrays directly to network
    # filesystems, by first saving to an in-memory IO buffer.
    np_path = file_path[:-5] + '_ess_{}.npz'.format(i)
    with tf.gfile.GFile(np_path, 'wb') as out_f:
      io_buffer = io.BytesIO()
      np.savez(io_buffer, **dict_ess)
      out_f.write(io_buffer.getvalue())

    txt_path = file_path[:-5] + '_ess_{}.txt'.format(i)
    with tf.gfile.GFile(txt_path, 'w') as out_f:
      for k, v in dict_ess.items():
        out_f.write('{}: {}\n\n'.format(k, v))

    np_path = file_path[:-5] + '{}.npz'.format(i)
    with tf.gfile.GFile(np_path, 'wb') as out_f:
      io_buffer = io.BytesIO()
      np.savez(io_buffer, **dict_res)
      out_f.write(io_buffer.getvalue())

if __name__ == '__main__':
  tf.app.run()
