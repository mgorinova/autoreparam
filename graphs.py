import collections
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import util as util
import program_transformations as ed_transforms

FLAGS = tf.app.flags.FLAGS


def make_cp_graph(model_config):
  """
                Constructs the CP graph of the given model.
                Resets the default TF graph.
        """

  tf.compat.v1.reset_default_graph()

  model = model_config.model
  if model_config.bijectors_fn is not None:
    model = ed_transforms.transform_with_bijectors(
        model, model_config.bijectors_fn)

  log_joint_centered = ed.make_log_joint_fn(model)

  with ed.tape() as model_tape:
    _ = model(*model_config.model_args)

  target_cp_kwargs = {}
  for param in model_tape.keys():
    if param in model_config.observed_data.keys():
      target_cp_kwargs[param] = model_config.observed_data[param]

  def target_cp(*param_args):
    i = 0
    for param in model_tape.keys():
      if param not in model_config.observed_data.keys():
        target_cp_kwargs[param] = param_args[i]
        i = i + 1

    return log_joint_centered(*model_config.model_args, **target_cp_kwargs)

  elbo, variational_parameters = util.get_mean_field_elbo(
      model,
      target_cp,
      num_mc_samples=FLAGS.num_mc_samples,
      model_args=model_config.model_args,
      model_obs_kwargs=model_config.observed_data,
      vi_kwargs=None)

  return target_cp, model_config.model, elbo, variational_parameters, None


def make_ncp_graph(model_config):
  """
                Constructs the CP graph of the given model.
                Resets the default TF graph.
        """
  tf.compat.v1.reset_default_graph()

  interceptor = ed_transforms.ncp

  def model_ncp(*params):
    with ed.interception(interceptor):
      return model_config.model(*params)

  if model_config.bijectors_fn is not None:
    model_ncp = ed_transforms.transform_with_bijectors(
        model_ncp, model_config.bijectors_fn)

  log_joint_noncentered = ed.make_log_joint_fn(model_ncp)

  with ed.tape() as model_tape:
    _ = model_ncp(*model_config.model_args)

  target_ncp_kwargs = {}
  for param in model_tape.keys():
    if param in model_config.observed_data.keys():
      target_ncp_kwargs[param] = model_config.observed_data[param]

  def target_ncp(*param_args):
    i = 0
    for param in model_tape.keys():
      if param not in model_config.observed_data.keys():
        target_ncp_kwargs[param] = param_args[i]
        i = i + 1

    return log_joint_noncentered(*model_config.model_args, **target_ncp_kwargs)

  elbo, variational_parameters = util.get_mean_field_elbo(
      model_config.model,
      target_ncp,
      num_mc_samples=FLAGS.num_mc_samples,
      model_args=model_config.model_args,
      model_obs_kwargs=model_config.observed_data,
      vi_kwargs=None)

  return target_ncp, model_ncp, elbo, variational_parameters, None


def make_cvip_graph(model_config,
                    parameterisation_type='exp',
                    tied_pparams=False):
  """
                Constructs the cVIP graph of the given model.
                Resets the default TF graph.
        """

  tf.compat.v1.reset_default_graph()

  results = collections.OrderedDict()

  (learnable_parameters, learnable_parametrisation,
   _) = ed_transforms.make_learnable_parametrisation(
       tau=1., parameterisation_type=parameterisation_type,
       tied_pparams=tied_pparams)

  def model_vip(*params):
    with ed.interception(learnable_parametrisation):
      return model_config.model(*params)

  if model_config.bijectors_fn is not None:
    model_vip = ed_transforms.transform_with_bijectors(
        model_vip, model_config.bijectors_fn)

  log_joint_vip = ed.make_log_joint_fn(model_vip)  # log_joint_fn

  with ed.tape() as model_tape:
    _ = model_vip(*model_config.model_args)

  target_vip_kwargs = {}
  for param in model_tape.keys():
    if param in model_config.observed_data.keys():
      target_vip_kwargs[param] = model_config.observed_data[param]

  def target_vip(*param_args):  # latent_log_joint_fn
    i = 0
    for param in model_tape.keys():
      if param not in model_config.observed_data.keys():
        target_vip_kwargs[param] = param_args[i]
        i = i + 1
    return log_joint_vip(*model_config.model_args, **target_vip_kwargs)

  #full_kwargs = collections.OrderedDict(model_config.observed_data.items())
  #full_kwargs['parameterisation'] = collections.OrderedDict()
  #for k in learnable_parameters.keys():
  #	full_kwargs['parameterisation'][k] = learnable_parameters[k]

  elbo, variational_parameters = util.get_mean_field_elbo(
      model_vip,
      target_vip,
      num_mc_samples=FLAGS.num_mc_samples,
      model_args=model_config.model_args,
      model_obs_kwargs=model_config.observed_data,
      vi_kwargs={'parameterisation':
          learnable_parameters})  #vi_kwargs=full_kwargs

  return target_vip, model_vip, elbo, variational_parameters, learnable_parameters


def make_dvip_graph(model_config, reparam, parameterisation_type='exp'):
  """
                Constructs the dVIP graph of the given model, where `reparam` is
                a cVIP
                reparameterisation obtained previously.
                Resets the default TF graph.
        """

  tf.compat.v1.reset_default_graph()

  results = collections.OrderedDict()

  _, insightful_parametrisation, _ = ed_transforms.make_learnable_parametrisation(
      learnable_parameters=reparam, parameterisation_type=parameterisation_type)

  def model_vip(*params):
    with ed.interception(insightful_parametrisation):
      return model_config.model(*params)

  if model_config.bijectors_fn is not None:
    model_vip = ed_transforms.transform_with_bijectors(
        model_vip, model_config.bijectors_fn)

  log_joint_vip = ed.make_log_joint_fn(model_vip)  # log_joint_fn

  with ed.tape() as model_tape:
    _ = model_vip(*model_config.model_args)

  target_vip_kwargs = {}
  for param in model_tape.keys():
    if param in model_config.observed_data.keys():
      target_vip_kwargs[param] = model_config.observed_data[param]

  def target_vip(*param_args):  # latent_log_joint_fn
    i = 0
    for param in model_tape.keys():
      if param not in model_config.observed_data.keys():
        target_vip_kwargs[param] = param_args[i]
        i = i + 1
    return log_joint_vip(*model_config.model_args, **target_vip_kwargs)

  elbo, variational_parameters = util.get_mean_field_elbo(
      model_vip,
      target_vip,
      num_mc_samples=FLAGS.num_mc_samples,
      model_args=model_config.model_args,
      model_obs_kwargs=model_config.observed_data,
      vi_kwargs={'parameterisation': reparam})

  return target_vip, model_vip, elbo, variational_parameters, None
