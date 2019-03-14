import numpy as np
import os
import json

import tensorflow
flags = tensorflow.app.flags

flags.DEFINE_boolean("elbos", default=False, help="")
flags.DEFINE_boolean("reparams", default=False, help="")
flags.DEFINE_string("model", default="all", help="")

FLAGS = flags.FLAGS

methods = [
    "CP",
    "NCP",
    "cVIP_exp",  #"cVIP_scale", 
    #"cVIP_exp_reparam_variational",
    "dVIP_exp",  #"dVIP_scale",  
    #"dVIP_exp_reparam_variational"
]

cvip_methods = ["cVIP_exp", "cVIP_scale", "cVIP_exp_reparam_variational"]


def get_elbos(model_name):

  elbos = {}
  stds = {}

  for m in methods:
    file_path = os.path.join(model_name, m + ".json")

    with tf.gfile.Open(file_path, "r") as f:
      results = json.load(f)
      elbos[m] = results["elbo"]
      stds[m] = results["estimated_elbo_std"]

  return elbos, stds


def get_reparam(model_name):

  discrete_parameterisation = {}

  for m in cvip_methods:
    file_path = os.path.join(model_name, m + ".json")

    with tf.gfile.Open(file_path, "r") as f:
      results = json.load(f)
      reparam = results["learned_reparam"]

    discrete_parameterisation[m] = dict(
        [(key, (np.array(reparam[key]) >= 0.5).astype(np.float32))
         for key in reparam.keys()])

  return discrete_parameterisation


model_names = [
    "8schools_", "german_credit_lognormalcentered_", "radon_MA", "radon_AZ",
    "radon_IN", "radon_MO", "radon_MN", "radon_PA", "radon_ND", "election_",
    "electric_", "time_series_"
]


def main(_):

  if (FLAGS.elbos):

    for name in (model_names if FLAGS.model == "all" else [FLAGS.model]):

      print(" ******  {}  ****** ".format(name))
      e, s = get_elbos(name)

      for key in e.keys():
        print("{0:.4f} +/- {1:.2f}   : {2}".format(e[key], s[key], key))

      print("\n\n")

  if (FLAGS.reparams):
    for name in (model_names if FLAGS.model == "all" else [FLAGS.model]):

      print(" ******  {}  ****** ".format(name))
      reparam = get_reparam(name)

      for m in cvip_methods:
        print("   {}".format(m))
        for k, v in reparam[m].items():
          print("{:>10}: {}".format(k, v))

        print("\n")


if __name__ == "__main__":
  tensorflow.app.run()
