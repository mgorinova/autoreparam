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
"""Interleaving Transition Kernel."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.mcmc import TransitionKernel

__all__ = [
    'Interleaved',
]

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.simplefilter('always')

InterleavedKernelResults = collections.namedtuple('InterleavedKernelResults',
                                                  ['cp_results', 'ncp_results'])


def noop(state):
  return state


def make_name(super_name, default_super_name, sub_name):
  """Helper which makes a `str` name; useful for tf.name_scope."""
  name = super_name if super_name is not None else default_super_name
  if sub_name is not None:
    name += '_' + sub_name
  return name


class Interleaved(TransitionKernel):

  def __init__(self,
               inner_kernel_cp,
               inner_kernel_ncp,
               to_cp=noop,
               to_ncp=noop,
               seed=None,
               name=None):

    self._seed_stream = tfp.distributions.SeedStream(seed,
                                                     'interleaved_one_step')


    # Support SimpleStepSizeAdaptation wrappers.
    _seed = lambda k: k.inner_kernel.seed if hasattr(k, 'inner_kernel'
                                                    ) else k.seed
    if (_seed(inner_kernel_cp) == _seed(inner_kernel_ncp) and
        _seed(inner_kernel_cp) is not None):
      raise Exception(
          'The two interleaved kernels cannot have the same random seed.')

    self._parameters = dict(
        inner_kernels={
            'cp': inner_kernel_cp,
            'ncp': inner_kernel_ncp
        },
        to_cp=to_cp,
        to_ncp=to_ncp,
        seed=seed,
        name=name)

  @property
  def to_cp(self):
    return self._parameters['to_cp']

  @property
  def to_ncp(self):
    return self._parameters['to_ncp']

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernels']

  @property
  def seed(self):
    return self._parameters['seed']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results):

    with tf.compat.v1.name_scope(
        name=make_name(self.name, 'iterleaved', 'one_step'),
        values=[current_state, previous_kernel_results]):

      # Recompute log prob at new state by bootstrapping
      cp_results = previous_kernel_results.cp_results
      new_inner_results = self.inner_kernel[
          'cp'].inner_kernel.bootstrap_results(current_state)
      cp_results = cp_results._replace(inner_results=new_inner_results)

      # Take a step in the CP space
      [next_cp_state, kernel_res_cp
      ] = self.inner_kernel['cp'].one_step(current_state, cp_results)

      current_ncp_state = self.to_ncp(next_cp_state)
      for i in range(len(current_ncp_state)):
        current_ncp_state[i] = tf.identity(current_ncp_state[i])

      # Recompute log prob at new state by bootstrapping
      ncp_results = previous_kernel_results.ncp_results
      new_inner_results = self.inner_kernel[
          'ncp'].inner_kernel.bootstrap_results(current_ncp_state)
      ncp_results = ncp_results._replace(inner_results=new_inner_results)

      # Take a step in the NCP space
      [
          next_ncp_state,
          kernel_res_ncp,
      ] = self.inner_kernel['ncp'].one_step(current_ncp_state, ncp_results)

      next_state = self.to_cp(next_ncp_state)

      for i in range(len(next_state)):
        next_state[i] = tf.identity(next_state[i])

      kernel_results = self.bootstrap_results(
          next_state, ncp_results=kernel_res_ncp, cp_results=kernel_res_cp)

      return next_state, kernel_results

  def bootstrap_results(self, init_state, ncp_results=None, cp_results=None):
    """Returns an object with the same type as returned by `one_step`."""
    with tf.compat.v1.name_scope(
        name=make_name(self.name, 'interleaved', 'bootstrap_results'),
        values=[init_state]):

      if ncp_results is None:
        ncp_results = self.inner_kernel['ncp'].bootstrap_results(
            self.to_ncp(init_state))
      if cp_results is None:
        cp_results = self.inner_kernel['cp'].bootstrap_results(init_state)
      return InterleavedKernelResults(
          ncp_results=ncp_results, cp_results=cp_results)
