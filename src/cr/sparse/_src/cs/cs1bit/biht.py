# Copyright 2021 CR-Suite Development Team
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


import jax.numpy as jnp
from jax import vmap, jit, lax
from jax.numpy.linalg import norm

from cr.nimble.dsp import (hard_threshold,
    build_signal_from_indices_and_values)


class BIHTState(NamedTuple):
    """Represents the state of the BIHT algorithm
    """
    # The non-zero values
    x_I: jnp.ndarray
    """Non-zero values"""
    I: jnp.ndarray
    """The support for non-zero values"""
    r: jnp.ndarray
    """The residuals"""
    n_mismatched_bits: jnp.ndarray
    # """The number of bits that are mismatched"""
    iterations: int
    """The number of iterations it took to complete"""


def biht(Phi, y, K, tau, max_iters=1000):
    r"""Solves the 1-bit compressive sensing problem :math:`\text{sgn} (\\Phi x) = y` using 
    Binary Iterative Hard Thresholding

    Args:
        Phi (jax.numpy.ndarray): A random dictionary of shape (M, N)
        y (jax.numpy.ndarray): The 1-bit measurements 
        K (int): Sparsity level of solution x (number of non-zero entries)
        tau (float): Step size for the x update step
        max_iters (int): Maximum number of iterations

    Returns:
        (BIHTState): A named tuple containing the solution x and other details


    We assume that :math:`x` is a K-sparse vector.

    We assume that the one-bit measurements are made as follows:

    .. math::

        y = \text{sgn} (\Phi x)

    Thus the vector y contains entries 1 and -1 for the signs of the entries in the
    measurement :math:`\Phi x`.


    The BIHT algorithm proceeds as follows:

    - Start with an estimate :math:`x = 0`
    - Compute the guess :math:`\hat{y} = \text{sgn} (\Phi x)`
    - Measure the residual :math:`r = y - \hat{y}`
    - Count the number of mismatched bits as number of places where r is non-zero.
    - Compute the correlation :math:`h = \Phi^T r`
    - Update x as :math:`x = x + \frac{\tau}{2} h`
    - Hard threshold x to keep only K largest entries
    - Repeat till convergence


    Example:
        >>> import cr.sparse as crs
        >>> import cr.sparse.dict as crdict
        >>> import cr.sparse.data as crdata
        >>> import cr.sparse.cs.cs1bit as cs1bit
        >>> M, N, K = 256, 512, 4
        >>> Phi = crdict.gaussian_mtx(cnb.KEYS[0], M, N, normalize_atoms=False)
        >>> x, omega = crdata.sparse_normal_representations(cnb.KEYS[1], N, K)
        >>> x = x / norm(x)
        >>> y = cs1bit.measure_1bit(Phi, x)
        >>> s0 = crdict.upper_frame_bound(Phi)
        >>> tau = 0.98 * s0
        >>> state = cs1bit.biht_jit(Phi, y, K, tau)
        >>> x_rec = build_signal_from_indices_and_values(N, state.I, state.x_I)
        >>> x_rec = x_rec / norm(x_rec)
    """
    ## Initialize some constants for the algorithm
    M, N = Phi.shape

    def init():
        # Data for the initial approximation [r = y, x = 0]
        I = jnp.arange(0, K)
        x_I = jnp.zeros(K)
        # Assume initial estimate to be zero and compute residual
        # compute the 1 bit output based on current x estimate
        y_int = Phi[:,I] @ x_I
        y_hat = jnp.sign(y_int)
        # difference between actual y and current estimate
        r = y - y_hat
        n_mismatched_bits = jnp.sum(r != 0)
        return BIHTState(x_I=x_I, I=I, r=r, n_mismatched_bits=n_mismatched_bits, 
            iterations=0)

    def body(state):
        # Compute the correlation of the residual with the atoms of Phi
        h = Phi.T @ state.r
        # current approximation
        x = build_signal_from_indices_and_values(N, state.I, state.x_I)
        # update
        x = x + tau/2 * h
        # threshold
        I, x_I = hard_threshold(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # compute the 1 bit output based on current x estimate
        y_int = Phi_I @ x_I
        y_hat = jnp.sign(y_int)
        # Compute new residual
        # difference between actual y and current estimate
        r = y - y_hat
        # Compute residual norm squared
        n_mismatched_bits = jnp.sum(r != 0)
        # update new state
        return BIHTState(x_I=x_I, I=I, r=r, n_mismatched_bits=n_mismatched_bits,
            iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        a = state.n_mismatched_bits > 0
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        # overall condition
        return c

    state = lax.while_loop(cond, body, init())
    return state


biht_jit = jit(biht, static_argnums=(2))
