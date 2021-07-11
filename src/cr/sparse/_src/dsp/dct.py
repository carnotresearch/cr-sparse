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

"""Discrete Cosine Transforms

Adapted from:

http://www-personal.umich.edu/~mejn/computational-physics/dcst.py
"""

from jax import jit
import jax.numpy as jnp
import jax.numpy.fft as jfft

def dct(y):
    """Computes the 1D Type-II DCT transform of y
    """
    n = y.shape[0]
    n2 = jnp.sqrt(2*n+2)
    y2 = jnp.concatenate( (y[:],  y[::-1]))
    c = jfft.rfft(y2, axis=0)
    phi = jnp.exp(-1j*jnp.pi*jnp.arange(n)/(2*n))
    prod = (phi*c[:n].T).T
    return jnp.real(prod)


def idct(a):
    """Computes the 1D Type-II IDCT transform of a
    """
    n = a.shape[0]
    phi = jnp.exp(1j*jnp.pi*jnp.arange(n)/(2*n))
    upper = (phi*a.T).T
    lower = jnp.zeros((1,)+a.shape[1:])
    c = jnp.concatenate((upper, lower))
    return jfft.irfft(c, axis=0)[:n]


def orthogonal_dct(y):
    """Computes the 1D Type-II DCT transform of y such that the transform is orthogonal
    """
    n = y.shape[0]
    n2 = jnp.sqrt(2*n+2)
    y2 = jnp.concatenate( (y[:],  y[::-1]))
    c = jfft.rfft(y2, axis=0)
    phi = jnp.exp(-1j*jnp.pi*jnp.arange(n)/(2*n))
    # see Wikipedia article on scaling to make the transform orthogonal
    phi = phi.at[0].set(phi[0]*1/jnp.sqrt(2))
    phi = phi*jnp.sqrt(2)/n 
    prod = (phi*c[:n].T).T
    return jnp.real(prod)

def orthogonal_idct(a):
    """Computes the 1D Type-II IDCT transform of a such that the transform is orthogonal
    """
    n = a.shape[0]
    phi = jnp.exp(1j*jnp.pi*jnp.arange(n)/(2*n))
    # see Wikipedia article on scaling to make the transform orthogonal
    phi = phi*n/jnp.sqrt(2) 
    phi = phi.at[0].set(phi[0]*jnp.sqrt(2))
    upper = (phi*a.T).T
    lower = jnp.zeros((1,)+a.shape[1:])
    c = jnp.concatenate((upper, lower))
    return jfft.irfft(c, axis=0)[:n]
