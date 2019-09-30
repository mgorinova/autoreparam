# coding=utf-8
# Copyright 2018 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model definitions."""
# pylint: disable=missing-docstring,g-doc-args,g-doc-return-or-yield
# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os
import numpy as np
import pandas as pd
from six.moves import urllib
import csv

import tensorflow as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from tensorflow_probability import edward2 as ed
import program_transformations as ed_transforms
from tensorflow_probability import positive_semidefinite_kernels as psd_kernels

import electric
import election88

data_dir = '/tmp/datasets'

ModelConfig = collections.namedtuple(
    'ModelConfig', ('model', 'model_args', 'observed_data', 'to_centered',
                    'to_noncentered', 'make_to_centered'))


def build_make_to_centered(model, model_args, observed_data={}):
  """Make a fn to convert a model's state to centered parameterisation."""

  def make_to_centered(**centering_kwargs):
    (_, parametrisation, _) = ed_transforms.make_learnable_parametrisation(
        learnable_parameters=centering_kwargs)

    def to_centered(uncentered_state):
      set_values = ed_transforms.make_value_setter(*uncentered_state)
      with ed.interception(set_values):
        with ed.interception(parametrisation):
          with ed.tape() as centered_tape:
            model(*model_args)

      param_vals = [
          tf.identity(v)
          for k, v in centered_tape.items()
          if k not in observed_data.keys()
      ]
      return param_vals
      # [tf.identity(v) for v in list(centered_tape.values())[:-1]]

    return to_centered

  return make_to_centered


def make_to_noncentered(model, model_args, observed_data={}):
  """Make a fn to convert a model's state to noncentered parameterisation."""

  def to_noncentered(centered_state):
    set_values = ed_transforms.make_value_setter(*centered_state)
    with ed.tape() as noncentered_tape:
      with ed.interception(ed_transforms.ncp):
        with ed.interception(set_values):
          model(*model_args)

    param_vals = [
        tf.identity(v)
        for k, v in noncentered_tape.items()
        if k not in observed_data.keys()
    ]
    return param_vals
    # [tf.identity(v) for v in list(noncentered_tape.values())[:-1]]

  return to_noncentered


def get_eight_schools():
  """Eight schools model."""
  num_schools = 8
  treatment_effects = np.array([28, 8, -3, 7, -1, 1, 18, 12],
                               dtype=np.float32)
  treatment_stddevs = np.array([15, 10, 16, 11, 9, 11, 10, 18],
                               dtype=np.float32)

  def schools_model(num_schools, stddevs):
    mu = ed.Normal(loc=0., scale=5., name='mu')
    log_tau = ed.Normal(loc=0., scale=5., name='log_tau')
    theta = ed.Normal(
        loc=mu * tf.ones(num_schools),
        scale=tf.exp(log_tau) * tf.ones(num_schools),
        name='theta')
    y = ed.Normal(loc=theta, scale=stddevs, name='y')
    return y

  model_args = [num_schools, treatment_stddevs]
  observed = {'y': treatment_effects}

  varnames = ['mu', 'log_tau', 'theta']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      schools_model, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      schools_model, model_args=model_args, observed_data=observed)

  return ModelConfig(schools_model, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_multivariate_simple():
  dim = 3
  scale = np.identity(3, dtype=np.float32)

  num_datapoints = 4
  data = np.array([[0., 1., 0.],
                   [-0.2, 1.1, 0.1],
                   [0.5, 1.2, 0.2],
                   [1., -0.2, -1.2]], dtype=np.float32)

  def multivariate_normal_model(num_datapoints):
    A = ed.Wishart(df=dim, scale=scale, name='A')
    x = ed.MultivariateNormalFullCovariance(
      loc=tf.zeros(dim), covariance_matrix=A, name='x')
    loc = tf.ones([num_datapoints, 1]) * x
    y = ed.Normal(loc=loc, scale=1., name='y')
    return y

  model_args = [num_datapoints]
  observed = {'y': data}

  varnames = ['V', 'x']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
    multivariate_normal_model, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
    multivariate_normal_model, model_args=model_args, observed_data=observed)

  return ModelConfig(multivariate_normal_model, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_gp_classification():
  X_train = np.array([[0.284, 0.399], [0.295, 0.422], [0.297, 0.410], [0.282, 0.408],
                      [0.299, 0.406], [0.287, 0.380], [0.290, 0.395], [0.281, 0.377],
                      [0.280, 0.374], [0.289, 0.391], [0.286, 0.369], [0.298, 0.351],
                      [0.295, 0.422], [0.296, 0.410], [0.286, 0.368], [0.287, 0.352],
                      [0.280, 0.374], [0.289, 0.391], [0.286, 0.369], [0.298, 0.351],
                      [0.295, 0.422], [0.288, 0.410], [0.286, 0.368], [0.287, 0.352],
                      [0.286, 0.361], [0.287, 0.351], [0.290, 0.390], [0.298, 0.389],
                      [0.283, 0.389], [0.284, 0.422], [0.297, 0.410], [0.282, 0.418],
                      [0.297, 0.406], [0.283, 0.381], [0.280, 0.395], [0.281, 0.367],
                      [0.290, 0.374], [0.284, 0.391], [0.286, 0.369], [0.288, 0.351],
                      [0.282, 0.339], [0.293, 0.337], [0.291, 0.392], [0.329, 0.286],
                      [0.323, 0.208], [0.368, -0.221], [0.398, 0.99], [0.659, 0.142],
                      [0.689, 0.442], [0.320, 0.356], [0.294, 0.432], [0.299, 0.382],
                      [0.287, 0.423], [0.289, 0.393], [0.292, 0.363], [0.291, 0.423]
                      ]).astype(np.float32)

  num_train_points, num_dims = X_train.shape

  y_lbls = [X_train[i, 0] > 0.3 for i in range(num_train_points)]

  def gp_classification_model(index_points):
    kernel = psd_kernels.ExponentiatedQuadratic(
      amplitude=100.,
      length_scale=1.)
    observation_noise_variance = 1.

    # Define a GP->Bernoulli model over the given index points.
    latent_logits = ed.GaussianProcess(
      kernel=kernel,
      index_points=index_points,
      observation_noise_variance=observation_noise_variance,
      name='latent_logits')

    y_labels = ed.Bernoulli(logits=latent_logits, name='y_labels')
    return y_labels

  model_args = [X_train]
  observed = {'y_labels': y_lbls}

  varnames = ['latent_logits']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
    gp_classification_model, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
    gp_classification_model, model_args=model_args, observed_data=observed)

  return ModelConfig(gp_classification_model, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_gp_poisson():

  with open('rongelap.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for first_row in reader:
      keys = first_row[1:5]
      dict = {k: [] for k in keys}
      break

    for row in reader:
      vals = row[1:5]
      for i in range(len(keys)):
        dict[keys[i]].append(vals[i])

  X_train = np.column_stack((dict['X1'], dict['X2'])).astype(np.float32)
  y_data = np.array(dict['data']).astype(np.int32)

  def gp_poisson_model(index_points):

    # kernel_log_lengthscale = ed.Normal(loc=0., scale=1., name='kernel_log_lengthscale')
    # kernel_log_amplitude = ed.Normal(loc=0., scale=1., name='kernel_log_amplitude')

    kernel = psd_kernels.ExponentiatedQuadratic(
      amplitude=1., #tf.exp(kernel_log_amplitude),
      length_scale=1.) #tf.exp(kernel_log_lengthscale))

    observation_noise_variance = 1.

    # Define a GP->Bernoulli model over the given index points.
    log_rate = ed.GaussianProcess(
      kernel=kernel,
      index_points=index_points,
      observation_noise_variance=observation_noise_variance,
      name='log_rate')

    y = ed.Poisson(rate=tf.exp(log_rate), name='y')
    return y

  model_args = [X_train]
  observed = {'y': y_data}

  varnames = [# 'kernel_log_amplitude',
              # 'kernel_log_lengthscale',
              'log_rate']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
    gp_poisson_model, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
    gp_poisson_model, model_args=model_args, observed_data=observed)

  return ModelConfig(gp_poisson_model, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_neals_funnel():
  """Neal's funnel."""

  def neals_funnel():
    x1 = ed.Normal(loc=0., scale=3., name='x1')
    x2 = ed.Normal(loc=0., scale=tf.exp(x1 / 2.), name='x2')
    return x1, x2

  model_args = []
  observed = {}

  varnames = ['x1', 'x2']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      neals_funnel, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      neals_funnel, model_args=model_args, observed_data=observed)

  return ModelConfig(neals_funnel, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


@contextlib.contextmanager
def open_from_url(url):
  filename = os.path.basename(url)
  path = os.path.expanduser(data_dir)
  filepath = os.path.join(path, filename)
  if not os.path.exists(filepath):
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    print('Downloading %s to %s' % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
  with open(filepath, 'r') as f:
    yield f


def load_radon_data(state_code):
  """Load the radon dataset.

  Code from http://mc-stan.org/users/documentation/case-studies/radon.html.
  (Apache2 licensed)
  """

  # Non-NaN possibilities: MN, IN, MO, ND, PA

  with open_from_url(
    'http://www.stat.columbia.edu/~gelman/arm/examples/radon/srrs2.dat') as f:
    srrs2 = pd.read_csv(f)
  srrs2.columns = srrs2.columns.map(str.strip)
  srrs_mn = srrs2.assign(fips=srrs2.stfips * 1000 +
                         srrs2.cntyfips)[srrs2.state == state_code]

  with open_from_url(
    'http://www.stat.columbia.edu/~gelman/arm/examples/radon/cty.dat') as f:
    cty = pd.read_csv(f)
  cty_mn = cty[cty.st == state_code].copy()
  cty_mn['fips'] = 1000 * cty_mn.stfips + cty_mn.ctfips

  srrs_mn.county = srrs_mn.county.str.strip()

  counties = srrs_mn[['county', 'fips']].drop_duplicates()
  county_map_uranium = {
      a: b for a, b in zip(counties['county'], range(len(counties['county'])))
  }
  uranium_levels = cty_mn.merge(counties, on='fips')['Uppm']

  srrs_mn_new = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
  srrs_mn_new = srrs_mn_new.drop_duplicates(subset='idnum')

  srrs_mn_new.county = srrs_mn_new.county.str.strip()
  mn_counties = srrs_mn_new.county.unique()
  county_lookup = dict(zip(mn_counties, range(len(mn_counties))))

  county = srrs_mn_new['county_code'] = srrs_mn_new.county.replace(
      county_lookup).values
  radon = srrs_mn_new.activity
  srrs_mn_new['log_radon'] = log_radon = np.log(radon + 0.1).values
  floor_measure = srrs_mn_new.floor.values

  n_county = srrs_mn_new.groupby('county')['idnum'].count()

  uranium = np.zeros(len(n_county), dtype=np.float32)
  for k, _ in county_lookup.items():
    uranium[county_lookup[k]] = uranium_levels[county_map_uranium[k]]

  uranium = [(np.log(ur) if ur > 0. else 0.) for ur in uranium]

  c = county  # length N (with J unique values)
  u = np.float32(uranium)  # length J
  x = np.float32(floor_measure)  # length N
  data = np.float32(log_radon).reshape(-1, 1)  # length N

  return c, u, x, data


def get_radon_model_stddvs(state_code='MN'):
  """Radon model with inferred scale parameters."""

  c, u, x, data = load_radon_data(state_code=state_code)
  N = data.shape[0]
  J = len(u)

  print('Radon N={}, J={}'.format(N, J))

  def radon(J, county, u, x):
    mua = ed.Normal(loc=0., scale=1., name='mua')  # scalar
    b1 = ed.Normal(loc=0., scale=1., name='b1')  # scalar
    b2 = ed.Normal(loc=0., scale=1., name='b2')  # scalar

    m_mu = mua + u * b1

    m = ed.Normal(loc=m_mu, scale=tf.ones(J), name='m')  # J

    C = tf.one_hot(county, J)  # shape (N, J)

    log_m_stddv = ed.Normal(
        loc=tf.zeros(J), scale=tf.ones(J), name='log_m_stddv')

    y_mu = tf.matmul(C, tf.expand_dims(m, 1)) + tf.expand_dims(x, 1) * b2
    y_stddv = tf.matmul(C, tf.expand_dims(tf.exp(log_m_stddv), 1))
    return ed.Normal(loc=y_mu, scale=y_stddv, name='y')

  model_args = [J, c, u, x]
  observed = {'y': data}

  varnames = ['mua', 'b1', 'b2', 'm', 'log_m_stddv']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      radon, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      radon, model_args=model_args, observed_data=observed)

  return ModelConfig(radon, model_args, observed, to_centered, to_noncentered,
                     make_to_centered)


def get_radon(state_code='MN'):
  """Build a model for the radon dataset.

  'N': len(log_radon),
  'J': len(n_county),
  'county': county+1, # Stan counts starting at 1
  'u': u,
  'x': floor_measure,
  'y': log_radon}
  """

  c, u, x, data = load_radon_data(state_code=state_code)
  N = data.shape[0]
  J = len(u)

  print('Radon N={}, J={}'.format(N, J))

  def radon(J, county, u, x, sigma_y):
    mua = ed.Normal(loc=0., scale=1., name='mua')  # scalar
    b1 = ed.Normal(loc=0., scale=1., name='b1')  # scalar
    b2 = ed.Normal(loc=0., scale=1., name='b2')  # scalar

    m_mu = mua + u * b1
    m = ed.Normal(loc=m_mu, scale=tf.ones(J), name='m')

    C = tf.one_hot(county, J)  # shape (N, J)

    y_mu = tf.matmul(C, tf.expand_dims(m, 1)) + tf.expand_dims(x, 1) * b2
    return ed.Normal(loc=y_mu, scale=sigma_y, name='y')

  sigma_y = 1.

  model_args = [J, c, u, x, sigma_y]
  observed = {'y': data}

  varnames = ['mua', 'b1', 'b2', 'm']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      radon, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      radon, model_args=model_args, observed_data=observed)

  return ModelConfig(radon, model_args, observed, to_centered, to_noncentered,
                     make_to_centered)


def load_german_credit_data():
  """Load the German credit dataset."""
  with open_from_url(
   'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'  # pylint: disable=line-too-long
  ) as f:
    data = pd.read_csv(f, delim_whitespace=True, header=None)

  def categorical_to_int(x):
    d = {u: i for i, u in enumerate(np.unique(x))}
    return np.array([d[i] for i in x])

  categoricals = []
  numericals = [np.ones([len(data)])]
  for column in data.columns[:-1]:
    column = data[column]
    if column.dtype == 'O':
      categoricals.append(categorical_to_int(column))
    else:
      numericals.append((column - column.mean()) / column.std())
  numericals = np.array(numericals).T
  status = np.array(data[20] == 1, dtype=np.int32)

  return numericals, categoricals, status


def get_german_credit_lognormalcentered():
  """German credit model with LogNormal priors on the coefficient scales."""
  numericals, categoricals, status = load_german_credit_data()

  def german_credit_model():
    x_numeric = tf.constant(numericals.astype(np.float32))
    x_categorical = [tf.one_hot(c, c.max() + 1) for c in categoricals]
    all_x = tf.concat([x_numeric] + x_categorical, 1)
    num_features = int(all_x.shape[1])

    overall_log_scale = ed.Normal(loc=0., scale=10., name='overall_log_scale')
    beta_log_scales = ed.Normal(
        loc=overall_log_scale,
        scale=tf.ones([num_features]),
        name='beta_log_scales')
    beta = ed.Normal(
        loc=tf.zeros([num_features]),
        scale=tf.exp(beta_log_scales),
        name='beta')
    logits = tf.einsum('nd,md->mn', all_x, beta[tf.newaxis, :])
    return ed.Bernoulli(logits=logits, name='y')

  observed = {'y': status[np.newaxis, Ellipsis]}
  model_args = []

  varnames = ['overall_log_scale', 'beta_log_scales', 'beta']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      german_credit_model, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      german_credit_model, model_args=model_args, observed_data=observed)

  return ModelConfig(german_credit_model, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_german_credit_gammascale():
  """German credit model with Gamma priors on the coefficient scales."""
  numericals, categoricals, status = load_german_credit_data()

  def german_credit_model():
    x_numeric = tf.constant(numericals.astype(np.float32))
    x_categorical = [tf.one_hot(c, c.max() + 1) for c in categoricals]
    all_x = tf.concat([x_numeric] + x_categorical, 1)
    num_features = int(all_x.shape[1])

    overall_log_scale = ed.Normal(loc=0., scale=10., name='overall_log_scale')
    beta_log_scales = ed.TransformedDistribution(
        tfd.Gamma(0.5 * tf.ones([num_features]), 0.5),
        bijector=tfb.Invert(tfb.Exp()),
        name='beta_log_scales')
    beta = ed.Normal(loc=tf.zeros([num_features]),
                     scale=tf.exp(overall_log_scale + beta_log_scales),
                     name='beta')
    logits = tf.einsum('nd,md->mn', all_x, beta[tf.newaxis, :])
    return ed.Bernoulli(logits=logits, name='y')

  observed = {'y': status[np.newaxis, Ellipsis]}
  model_args = []

  varnames = ['overall_log_scale', 'beta_log_scales', 'beta']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      german_credit_model, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      german_credit_model, model_args=model_args, observed_data=observed)

  return ModelConfig(german_credit_model, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_election():

  def election(N, n_state, black, female, state):
    mua = ed.Normal(loc=0., scale=100., name='mua')

    log_sigma_a = ed.Normal(loc=0., scale=10., name='log_sigma_a')
    a = ed.Normal(loc=mua, scale=tf.ones([n_state]) * tf.exp(log_sigma_a), name='a')

    b1 = ed.Normal(loc=0., scale=100., name='b1')
    b2 = ed.Normal(loc=0., scale=100., name='b2')

    C = tf.one_hot(state, n_state)  # shape (N, J)

    y_hat = tf.matmul(C, tf.expand_dims(
        a, 1)) + tf.expand_dims(female, 1) * b2 + tf.expand_dims(black, 1) * b1
    return ed.Bernoulli(logits=y_hat, name='y')

  N = election88.data['N']
  n_state = election88.data['n_state']
  black = election88.data['black']
  female = election88.data['female']
  state = election88.data['state']
  y = np.reshape(election88.data['y'], (-1, 1))

  observed = {'y': y}
  model_args = [N, n_state, black, female, state]

  varnames = ['mua', 'log_sigma_a', 'a', 'b1', 'b2']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      election, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      election, model_args=model_args, observed_data=observed)

  return ModelConfig(election, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_electric():

  def electric_model(N, n_pair, n_grade, n_grade_pair, grade, grade_pair, pair,
                     treatment):

    C_pair = tf.one_hot(pair, n_pair)  # 192 x 96
    C_grade = tf.one_hot(grade, n_grade)  # 192 x 4
    C_grade_pair = tf.one_hot(grade_pair, n_grade_pair)  # 96 x 4

    mua = ed.Normal(loc=0., scale=tf.ones(n_grade_pair), name='mua')  # 4
    mua_hat = 100 * tf.matmul(C_grade_pair, mua[..., tf.newaxis])  # 96

    sigma_y = ed.Normal(loc=0., scale=tf.ones(n_grade), name='sigma_y')  # 4
    sigma_y_hat = tf.matmul(C_grade, sigma_y[..., tf.newaxis])  # 192

    a = ed.Normal(loc=mua_hat, scale=1., name='a')  # 96
    b = ed.Normal(loc=0., scale=100 * tf.ones(n_grade), name='b')  # 4

    y_hat_a = tf.reshape(tf.matmul(C_pair, a), [N])
    y_hat_b = tf.reshape(tf.matmul(C_grade, b[..., tf.newaxis]), [N])
    y_hat_sigma = tf.exp(tf.reshape(sigma_y_hat, [N]))

    y_hat = y_hat_a + tf.multiply(y_hat_b, treatment)

    return ed.Normal(loc=y_hat, scale=y_hat_sigma, name='y')

  N = electric.data['N']
  n_pair = electric.data['n_pair']
  n_grade = electric.data['n_grade']
  n_grade_pair = electric.data['n_grade_pair']
  grade = electric.data['grade']
  grade_pair = electric.data['grade_pair']
  pair = electric.data['pair']
  treatment = electric.data['treatment']
  y = electric.data['y']

  observed = {'y': y}
  model_args = [
      N, n_pair, n_grade, n_grade_pair, grade, grade_pair, pair, treatment
  ]

  varnames = ['mua', 'a', 'b', 'sigma_y']
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      electric_model, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      electric_model, model_args=model_args, observed_data=observed)

  return ModelConfig(electric_model, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)


def get_time_series():

  def time_series(T, x):

    sigma_alpha = ed.Normal(loc=0., scale=1., name='sigma_alpha')
    sigma_mu = ed.Normal(loc=0., scale=1., name='sigma_mu')
    # sigma_y = ed.Normal(loc=0.,scale=1., name="sigma_y")

    alpha = [
        ed.Normal(loc=0., scale=tf.nn.softplus(sigma_alpha), name='alpha0')
    ]
    mu = [ed.Normal(loc=0., scale=tf.nn.softplus(sigma_mu), name='mu0')]

    for t in range(1, T):
      alpha.append(
          ed.Normal(
              loc=alpha[t - 1] + mu[t - 1],
              scale=tf.nn.softplus(sigma_alpha),
              name='alpha{}'.format(t)))
      mu.append(
          ed.Normal(
              loc=mu[t - 1],
              scale=tf.nn.softplus(sigma_mu),
              name='mu{}'.format(t)))

    beta = ed.Normal(loc=0., scale=1., name='beta')

    return ed.Normal(loc=tf.stack(alpha) + beta * x, scale=0.12, name='y')

  x = [
      1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970,
      1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982,
      1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994,
      1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
      2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018
  ]
  y = [
      315.97, 316.91, 317.64, 318.45, 318.99, 319.62, 320.04, 321.38, 322.16,
      323.04, 324.62, 325.68, 326.32, 327.45, 329.68, 330.18, 331.11, 332.04,
      333.83, 335.4, 336.84, 338.75, 340.11, 341.45, 343.05, 344.65, 346.12,
      347.42, 349.19, 351.57, 353.12, 354.39, 355.61, 356.45, 357.1, 358.83,
      360.82, 362.61, 363.73, 366.7, 368.38, 369.55, 371.14, 373.28, 375.8,
      377.52, 379.8, 381.9, 383.79, 385.6, 387.43, 389.9, 391.65, 393.85,
      396.52, 398.65, 400.83, 404.24, 406.55, 408.52
  ]

  # x = x[-5:]
  # y = y[-5:]

  T = len(x)

  observed = {'y': y}
  model_args = [T, x]

  varnames = [
      'beta',
      'sigma_alpha',
      'sigma_mu',  # 'sigma_y'
  ] + ['alpha{}'.format(i) for i in range(T)] + ['mu{}'.format(i) for i in range(T)]
  param_names = [p for v in varnames for p in (v + '_a', v + '_b')]
  noncentered_parameterization = {p: 0. for p in param_names}

  make_to_centered = build_make_to_centered(
      time_series, model_args=model_args, observed_data=observed)
  to_centered = make_to_centered(**noncentered_parameterization)
  to_noncentered = make_to_noncentered(
      time_series, model_args=model_args, observed_data=observed)

  return ModelConfig(time_series, model_args, observed, to_centered,
                     to_noncentered, make_to_centered)
