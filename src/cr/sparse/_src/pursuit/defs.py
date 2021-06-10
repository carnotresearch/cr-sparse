# Copyright 2021 CR.Sparse Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple, List, Dict
from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

@dataclass
class SingleRecoverySolution:
    signals: jnp.DeviceArray = None
    representations : jnp.DeviceArray = None
    residuals : jnp.DeviceArray =  None
    residual_norms : jnp.DeviceArray = None
    iterations: int = None
    support : jnp.DeviceArray = None

class RecoverySolution(NamedTuple):
    """Represents the solution of a sparse recovery problem

    Consider a sparse recovery problem :math:`y=\Phi x + e`.
    Assume that :math:`x` is supported on an index set :math:`I`
    i.e. the non-zero values of :math:`x` are in the sub-vector
    :math:`x_I`, then the equation can be rewritten as 
    :math:`y = \Phi_I x_I + e`.

    Solving the sparse recovery problem given :math:`\Phi`
    and :math:`x` involves identifying :math:`I` and estimating :math:`x_I`.
    Then, the residual is :math:`r = y - \Phi_I x_I`. An important
    quantity during the sparse recovery is the (squared) norm of the
    residual :math:`\| r \|_2^2` which is an estimate of the energy
    of error :math:`e`.

    This type combines all of this information together.

    Parameters:

        x_I : :estimate(s) of :math:`x_I`
        I : identified index set(s) :math:`I`
        r : residual(s) :math:`r = y - \Phi_I x_I `
        r_norm_sqr: squared norm of residual :math:`\| r \|_2^2`
        iterations: Number of iterations required for the algorithm to converge

    Note:

        The tuple can be used to solve multiple measurement vector
        problems also. In this case, each column (of individual parameters)
        represents the solution of corresponding single vector problems.
    """
    # The non-zero values
    x_I: jnp.ndarray
    """Non-zero values"""

    I: jnp.ndarray
    """The support for non-zero values"""

    r: jnp.ndarray
    """The residuals"""
    r_norm_sqr: jnp.ndarray
    """The residual norm squared"""

    iterations: int
    """The number of iterations it took to complete"""


class PTConfig(NamedTuple):
    K: int
    M: int
    eta: int
    rho: int
    
class PTConfigurations(NamedTuple):
    N: int
    configurations: List[PTConfig]
    Ms: jnp.DeviceArray
    etas: jnp.DeviceArray
    rhos: jnp.DeviceArray
    reverse_map: Dict


class HTPState(NamedTuple):
    # The non-zero values
    x_I: jnp.ndarray
    """Non-zero values"""
    I: jnp.ndarray
    """The support for non-zero values"""
    r: jnp.ndarray
    """The residuals"""
    r_norm_sqr: jnp.ndarray
    """The residual norm squared"""
    iterations: int
    """The number of iterations it took to complete"""
    # Information from previous iteration
    I_prev: jnp.ndarray
    x_I_prev: jnp.ndarray
    r_norm_sqr_prev: jnp.ndarray

IHTState = HTPState

CoSaMPState = HTPState