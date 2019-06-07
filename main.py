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

import inference
import graphs
import models
import util

import program_transformations as ed_transforms

from tensorflow_probability.python import mcmc
from tensorflow.python.ops.parallel_for import pfor

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

flags.DEFINE_boolean(
    'tied_pparams',
    default=False,
    help='Whether to tie the loc and scale parameterisation parameters '
         'together as a single parameter (a=b)'
    )

flags.DEFINE_string('results_dir', default='', help='File to write results.')

flags.DEFINE_list(
    'learning_rates',
    default=[0.02, 0.05, 0.1, 0.2, 0.4],
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
    results_dir, 'cVIP_{}{}{}{}.json'.format(
        FLAGS.learnable_parmeterisation_type,
        '_tied' if FLAGS.tied_pparams else '',
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
           parameterisation_type=FLAGS.learnable_parmeterisation_type,
           tied_pparams=FLAGS.tied_pparams)
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


def tune_leapfrog_steps(model_config, num_tuning_samples, initial_step_size,
                        initial_states, results_dir):
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

    (states_orig, kernel_results, states, ess) = \
      inference.hmc(target, model, model_config,
                    initial_step_size,
                    initial_states=initial_states,
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
    util.print('Warning! This model contains Gamma variables, which '
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

  filename = '{}{}{}{}{}.json'.format(
      FLAGS.method,
      ('_' +
       FLAGS.learnable_parmeterisation_type if 'VIP' in FLAGS.method else ''),
      ('_tied'
       if FLAGS.tied_pparams else ''),
      ('_reparam_variational'
       if 'VIP' in FLAGS.method and FLAGS.reparameterise_variational else ''),
      ('_discrete_prior'
       if 'VIP' in FLAGS.method and FLAGS.discrete_prior else ''))

  file_path = os.path.join(results_dir, filename)

  if FLAGS.inference == 'VI':

    run_vi(model_config, results_dir, file_path)

  elif FLAGS.inference == 'HMC':
    if FLAGS.method == 'i':
      run_interleaved_hmc(model_config, results_dir, file_path)
    else:
      run_hmc(model_config, results_dir, file_path)


def run_vi(model_config, results_dir, file_path):
  (target, model, elbo, variational_parameters, learnable_parameters,
   actual_reparam) = create_target_graph(model_config, results_dir)

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
        tfp.distributions.Categorical(logits=[0., 5., 0.]), [
            tfp.distributions.Laplace(loc=0., scale=0.1),
            tfp.distributions.Uniform(),
            tfp.distributions.Laplace(loc=1., scale=0.1)
        ])

  start_time = time.time()
  (elbo_final, elbo_timeline, learning_rate, initial_step_size,
   learned_variational_params,
   learned_reparam) = inference.find_best_learning_rate(
       elbo,
       variational_parameters,
       learnable_parameters_prior=learnable_parameters_prior,
       learnable_parameters=learnable_parameters)
  end_time = time.time()

  # Save actual parameters used for dVIP
  if learned_reparam is None and isinstance(actual_reparam, dict):
    learned_reparam = actual_reparam

  def clean_dict(d):
    if d is None:
      return None
    else:
      return OrderedDict([(k,
                           d[k].item() if np.isscalar(d[k]) else d[k].tolist())
                          for k in d.keys()])

  results = {
   'elbo': elbo_final.item(),
   'variational_fit_time_secs': end_time-start_time,
   'actual_num_variational_steps': len(elbo_timeline),
   'estimated_elbo_std': (np.std(elbo_timeline[-32:])).item(),
   'learning_rate': learning_rate,
   'initial_step_size': [i.item() if np.isscalar(i) else i.tolist() \
              for i in initial_step_size],
   'learned_reparam': clean_dict(learned_reparam),
   'learned_variational_params': clean_dict(learned_variational_params),
  }

  with tf.gfile.Open(file_path, 'w') as outfile:
    json.dump(results, outfile)


def run_hmc(model_config, results_dir, file_path):
  if tf.gfile.Exists(file_path):
    with tf.gfile.Open(file_path, 'r') as f:
      prev_results = json.load(f)
  else:
    raise Exception('Run VI first to find initial step sizes')

  with ed.tape() as model_tape:
    model_config.model(*model_config.model_args)
  param_names = [
      k for k in list(model_tape.keys()) if k not in model_config.observed_data
  ]

  initial_step_size = prev_results['initial_step_size']
  num_ls = prev_results.get('num_leapfrog_steps', FLAGS.num_leapfrog_steps)
  initial_states = util.variational_inits_from_params(
      prev_results['learned_variational_params'],
      param_names=param_names,
      num_inits=FLAGS.num_chains).values()

  if num_ls is None and FLAGS.num_leapfrog_steps is None:
    num_ls = tune_leapfrog_steps(
        model_config,
        FLAGS.num_tuning_samples,
        initial_step_size,
        initial_states=initial_states,
        results_dir=results_dir)
  FLAGS.num_leapfrog_steps = num_ls
  with tf.gfile.Open(file_path, 'w') as f:
    prev_results['num_leapfrog_steps'] = FLAGS.num_leapfrog_steps
    json.dump(prev_results, f)

  util.print('\nNumber of leaprog steps is set to {}.\n'.format(
      FLAGS.num_leapfrog_steps))

  (target, model, elbo, variational_parameters, learnable_parameters,
   actual_reparam) = create_target_graph(model_config, results_dir)

  (states_orig, kernel_results, states, ess) = \
    inference.hmc(target, model, model_config, initial_step_size,
                  initial_states=initial_states,
                  reparam=(actual_reparam
                           if actual_reparam is not None
                           else learned_reparam))

  init = tf.global_variables_initializer()
  with tf.Session() as sess:

    init.run()

    start_time = time.time()

    samples, is_accepted, ess_final, samples_orig = sess.run(
        (states, kernel_results.inner_results.is_accepted, ess, states_orig))

    mcmc_time = time.time() - start_time

  normalized_ess_final = []
  for ess_ in ess_final:
    # report effective samples per 1000 gradient evals
    normalized_ess_final.append(1000 * ess_ /
                                (FLAGS.num_samples * FLAGS.num_leapfrog_steps))
  del ess_final

  ess_min, sem_min = util.get_min_ess(normalized_ess_final)
  util.print('ESS: {} +/- {}'.format(ess_min, sem_min))

  acceptance_rate = (
      np.sum(is_accepted) * 100. / float(FLAGS.num_samples * FLAGS.num_chains))

  save_hmc_results(
      prev_results,
      file_path=file_path,
      ess_min=ess_min.item(),
      sem_min=sem_min.item(),
      acceptance_rate=acceptance_rate.item(),
      mcmc_time_sec=mcmc_time)

  save_ess(
      file_path_base=file_path[:-5],
      samples=samples,
      param_names=param_names,
      normalized_ess_final=normalized_ess_final)


def run_interleaved_hmc_with_leapfrog_steps(
    model_config, results_dir, num_leapfrog_steps_cp, num_leapfrog_steps_ncp,
    initial_step_size_cp, initial_step_size_ncp, initial_states_cp):

  (target, model, elbo, variational_parameters, learnable_parameters,
   actual_reparam) = create_target_graph(model_config, results_dir)

  target_cp, target_ncp = target

  (states, kernel_results, ess) = \
   inference.hmc_interleaved(model_config, target_cp, target_ncp,
                             num_leapfrog_steps_cp=num_leapfrog_steps_cp,
                             num_leapfrog_steps_ncp=num_leapfrog_steps_ncp,
                             step_size_cp=initial_step_size_cp,
                             step_size_ncp=initial_step_size_ncp,
                             initial_states_cp=initial_states_cp,)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:

    init.run()

    start_time = time.time()

    cp_accepted = kernel_results.cp_results.inner_results.is_accepted
    ncp_accepted = kernel_results.ncp_results.inner_results.is_accepted
    samples, is_accepted_cp, is_accepted_ncp, ess_final = sess.run(
        (states, cp_accepted, ncp_accepted, ess))

    mcmc_time = time.time() - start_time

  normalized_ess_final = []
  for ess_ in ess_final:
    # report effective samples per 1000 gradient evals
    normalized_ess_final.append(1000 * ess_ /
                                (FLAGS.num_samples * FLAGS.num_leapfrog_steps))
  del ess_final

  ess_min, sem_min = util.get_min_ess(normalized_ess_final)
  util.print('ESS: {} +/- {}'.format(ess_min, sem_min))

  acceptance_rate_cp = (
      np.sum(is_accepted_cp) * 100. /
      float(FLAGS.num_samples * FLAGS.num_chains))
  acceptance_rate_ncp = (
      np.sum(is_accepted_ncp) * 100. /
      float(FLAGS.num_samples * FLAGS.num_chains))

  return (ess_min, sem_min, acceptance_rate_cp, acceptance_rate_ncp, mcmc_time,
          samples, normalized_ess_final)


def run_interleaved_hmc(model_config, results_dir, file_path):
  filename_cp = 'CP.json'
  filename_ncp = 'NCP.json'

  file_path_cp = os.path.join(results_dir, filename_cp)
  file_path_ncp = os.path.join(results_dir, filename_ncp)

  with ed.tape() as model_tape:
    model_config.model(*model_config.model_args)
  param_names = [
      k for k in list(model_tape.keys()) if k not in model_config.observed_data
  ]

  if tf.gfile.Exists(file_path_cp) and tf.gfile.Exists(file_path_ncp):
    with tf.gfile.Open(file_path_cp, 'r') as f:
      prev_results = json.load(f)
      initial_step_size_cp = prev_results['initial_step_size']
      num_leapfrog_steps_cp = prev_results['num_leapfrog_steps']
      learned_variational_params_cp = prev_results['learned_variational_params']

    with tf.gfile.Open(file_path_ncp, 'r') as f:
      prev_results = json.load(f)
      initial_step_size_ncp = prev_results['initial_step_size']
      num_leapfrog_steps_ncp = prev_results['num_leapfrog_steps']
  else:
    raise Exception('Run VI first to find initial step sizes, and HMC'
                    'first to find num_leapfrog_steps.')

  initial_states_cp = util.variational_inits_from_params(
      learned_variational_params_cp,
      param_names=param_names,
      num_inits=FLAGS.num_chains).values()

  best_ess_min = 0
  best_num_ls = None
  results = ()
  for num_ls in set([num_leapfrog_steps_ncp, num_leapfrog_steps_cp]):
    util.print('\nNumber of leaprog steps is set to {}.\n'.format(num_ls))
    FLAGS.num_leapfrog_steps = num_ls + num_ls
    (ess_min, sem_min, acceptance_rate_cp, acceptance_rate_ncp, mcmc_time,
     samples, normalized_ess_final) = run_interleaved_hmc_with_leapfrog_steps(
         model_config=model_config,
         results_dir=results_dir,
         num_leapfrog_steps_cp=num_ls,
         num_leapfrog_steps_ncp=num_ls,
         initial_step_size_cp=initial_step_size_cp,
         initial_step_size_ncp=initial_step_size_ncp,
         initial_states_cp=initial_states_cp)
    if ess_min.item() > best_ess_min:
      best_ess_min = ess_min.item()
      best_num_ls = num_ls
      results = (ess_min, sem_min, acceptance_rate_cp, acceptance_rate_ncp,
                 mcmc_time, samples, normalized_ess_final)
  (ess_min, sem_min, acceptance_rate_cp, acceptance_rate_ncp, mcmc_time,
   samples, normalized_ess_final) = results
  FLAGS.num_leapfrog_steps = best_num_ls + best_num_ls

  prev_results = {
      'initial_step_size_ncp': initial_step_size_ncp,
      'initial_step_size_cp': initial_step_size_cp,
      'num_leapfrog_steps': best_num_ls
  }

  save_hmc_results(
      prev_results,
      file_path=file_path,
      ess_min=ess_min.item(),
      sem_min=sem_min.item(),
      acceptance_rate_cp=acceptance_rate_cp.item(),
      acceptance_rate_ncp=acceptance_rate_ncp.item(),
      mcmc_time_sec=mcmc_time)

  save_ess(
      file_path_base=file_path[:-5],
      samples=samples,
      param_names=param_names,
      normalized_ess_final=normalized_ess_final)


def save_hmc_results(results, file_path, **kwargs):

  def init_results(list):
    for l in list:
      if l not in results.keys():
        results[l] = []

  init_results(kwargs.keys())

  for k, v in kwargs.items():
    results.get(k).append(v)  # TODO .item()

  with tf.gfile.Open(file_path, 'w') as outfile:
    json.dump(results, outfile)


def save_ess(file_path_base,
             samples,
             normalized_ess_final,
             param_names,
             num_chains_to_save=0):

  dict_ess = dict([(param_names[i], np.array(normalized_ess_final[i]))
                   for i in range(len(param_names))])

  # Work around issues saving np arrays directly to network
  # filesystems, by first saving to an in-memory IO buffer.
  np_path = file_path_base + '_ess_{}.npz'.format(i)
  with tf.gfile.GFile(np_path, 'wb') as out_f:
    io_buffer = io.BytesIO()
    np.savez(io_buffer, **dict_ess)
    out_f.write(io_buffer.getvalue())

  txt_path = file_path_base + '_ess_{}.txt'.format(i)
  with tf.gfile.GFile(txt_path, 'w') as out_f:
    for k, v in dict_ess.items():
      out_f.write('{}: {}\n\n'.format(k, v))
    out_f.write('\n\n')
    for k, v in dict_ess.items():
      out_f.write('{} mean: {}\n'.format(k, np.mean(v, axis=0)))
      out_f.write('{} stddev: {}\n\n'.format(k, np.std(v, axis=0)))

  if num_chains_to_save > 0:
    dict_res = dict([(param_names[i], samples[i][:, :num_chains_to_save]) for i in range(len(param_names))])
    np_path = file_path_base + '{}.npz'.format(i)
    with tf.gfile.GFile(np_path, 'wb') as out_f:
      io_buffer = io.BytesIO()
      np.savez(io_buffer, **dict_res)
      out_f.write(io_buffer.getvalue())

if __name__ == '__main__':
  tf.app.run()
