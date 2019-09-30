import numpy as np
import os
import json

import tensorflow as tf
flags = tf.app.flags

flags.DEFINE_boolean("elbos", default=False, help="")
flags.DEFINE_boolean("ess", default=False, help="")
flags.DEFINE_boolean("reparams", default=False, help="")
flags.DEFINE_boolean("normalize_times", default=False, help="")
flags.DEFINE_string("model", default="all", help="")
flags.DEFINE_string('results_dir', default='', help='')

FLAGS = flags.FLAGS

methods = [
    "CP",
    "NCP",
    "cVIP_exp",
    "cVIP_exp_tied"
]

cvip_methods = ["cVIP_exp_tied"]



def get_ess(model_name):

  ess = {}
  sem = {}
  leapfrog_steps = {}
  vi_times = {}
  mcmc_times = {}

  for m in methods + ["i"]:
    try:
      file_path = os.path.join(FLAGS.results_dir, model_name, m + ".json")

      with tf.io.gfile.GFile(file_path, "r") as f:
        results = json.load(f)
        ess[m] = results["ess_min"]
        sem[m] = results["sem_min"]
        leapfrog_steps[m] = (
            results["num_leapfrog_steps"] if "num_leapfrog_steps" in results
            else results["num_leapfrog_steps_cp"])
        vi_times[m] = results.get("variational_fit_time_secs", None)
        mcmc_times[m] = results["mcmc_time_sec"]
    except Exception as e:
      print(e)
      continue

  return ess, sem, leapfrog_steps, vi_times, mcmc_times


def get_elbos(model_name):

  elbos = {}
  stds = {}

  for m in methods:
    file_path = os.path.join(FLAGS.results_dir, model_name, m + ".json")

    with tf.io.gfile.GFile(file_path, "r") as f:
      results = json.load(f)
      elbos[m] = results["elbo"]
      stds[m] = results["estimated_elbo_std"]

  return elbos, stds


def get_reparam(model_name):

  discrete_parameterisation = {}
  parameterisation = {}

  for m in cvip_methods:
    file_path = os.path.join(FLAGS.results_dir, model_name, m + ".json")

    with tf.io.gfile.GFile(file_path, "r") as f:
      results = json.load(f)
      reparam = results["learned_reparam"]

    parameterisation[m] = dict([(key,
                                 (np.array(reparam[key])).astype(np.float32))
                                for key in reparam.keys()])
    discrete_parameterisation[m] = dict(
        [(key, (np.array(reparam[key]) >= 0.5).astype(np.float32))
         for key in reparam.keys()])

  return parameterisation


model_names = [
    "8schools_data", "german_credit_lognormalcentered_data", "radon_MA",
    "radon_AZ", "radon_IN", "radon_MO", "radon_MN", "radon_PA", "radon_ND",
    "radon_stddvs_MA", "radon_stddvs_AZ", "radon_stddvs_IN", "radon_stddvs_MO",
    "radon_stddvs_MN", "radon_stddvs_PA", "radon_stddvs_ND", "election_data",
    "electric_data", "time_series_data", 'election_data'
]


def main(_):

  if (FLAGS.elbos):

    for name in (model_names if FLAGS.model == "all" else [FLAGS.model]):

      print(" ******  {}  ****** ".format(name))

      try:
        e, s = get_elbos(name)
      except Exception as e:
        print(e)
        continue

      for key in e.keys():
        print("{0:.4f} +/- {1:.2f}   : {2}".format(e[key], s[key], key))

      print("\n\n")

  if (FLAGS.reparams):
    for name in (model_names if FLAGS.model == "all" else [FLAGS.model]):

      print(" ******  {}  ****** ".format(name))
      try:
        reparam = get_reparam(name)
      except Exception as e:
        print(e)
        continue

      for m in cvip_methods:
        print("   {}".format(m))
        for k, v in reparam[m].items():
          print("{:>10}: {}".format(k, v))

        print("\n")

  if (FLAGS.ess):
    for name in (model_names if FLAGS.model == "all" else [FLAGS.model]):

      print(" ******  {}  ****** ".format(name))
      try:
        ess, sem, leapfrog_steps, vi_times, mcmc_times = get_ess(name)
      except Exception as e:
        print(e)
        continue

      for key in ess.keys():
        if FLAGS.normalize_times:
          mcmc_time = mcmc_times[key][0]
          my_ess = ess[key][0]
          my_sem = sem[key][0]

          if key == "i":
            leapfrog_steps_per_sample = leapfrog_steps[key] * 2
            vi_time = vi_times["CP"] + vi_times["NCP"]
          else:
            leapfrog_steps_per_sample = leapfrog_steps[key]
            vi_time = vi_times[key]

          total_time = vi_time + mcmc_time
          total_grad_evals = leapfrog_steps_per_sample * 10000.
          total_effective_samples = my_ess * total_grad_evals / 1000.
          sampling_stderr = my_sem * total_grad_evals / 1000.

          time_per_variational_step = vi_time / 3000.
          time_per_variational_step_cp = vi_times["CP"] / 3000.

          time_per_step = (
              mcmc_time / (10000 * leapfrog_steps[key]) if key != 'i' else
              mcmc_time / (10000 * 2 * leapfrog_steps[key]))
          time_per_step_cp = mcmc_times["CP"][0] / (10000 * leapfrog_steps["CP"])
          print("{} +/- {} in {}s ({}s VI + {}s MCMC): {} ({} leapfrog steps, {:.2f}x/{:.2f}x CP time per VI/MCMC step)"
                .format(total_effective_samples, sampling_stderr, total_time,
                        vi_time, mcmc_time, key, leapfrog_steps[key],
                        time_per_variational_step/time_per_variational_step_cp,
                        time_per_step/time_per_step_cp))
        else:
          print("{} +/- {} : {} ({} leapfrog steps)".format(
              ess[key], sem[key], key, leapfrog_steps[key]))

      print("\n\n")


if __name__ == "__main__":
  tf.compat.v1.app.run()
