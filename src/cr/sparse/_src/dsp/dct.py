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

* http://www-personal.umich.edu/~mejn/computational-physics/dcst.py
* https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
"""

from jax import jit
import jax.numpy as jnp
import jax.numpy.fft as jfft

def dct(y):
    """Computes the 1D Type-II DCT transform

    Args:
        y (jax.numpy.ndarray): The 1D real signal

    Returns:
        jax.numpy.ndarray: The Type-II Discrete Cosine Transform coefficients of y
    """
    n = y.shape[0]
    y2 = jnp.concatenate( (y[:],  y[::-1]))
    c = jfft.rfft(y2, axis=0)[:n]
    ks = jnp.arange(n)
    phi = jnp.exp(-1j*jnp.pi*ks/(2*n))
    prod = (phi*c.T).T
    return jnp.real(prod)


def idct(a):
    """Computes the 1D Type-II Inverse DCT transform

    Args:
        a (jax.numpy.ndarray): The Type-II DCT transform coefficients of a 1D real signal

    Returns:
        jax.numpy.ndarray: The 1D real signal y s.t. a = dct(y)
     """
    n = a.shape[0]
    shape = (1,)+a.shape[1:]
    ks = jnp.arange(n)
    phi = jnp.exp(1j*jnp.pi*ks/(2*n))
    upper = (phi*a.T).T
    lower = jnp.zeros(shape)
    c = jnp.concatenate((upper, lower))
    return jfft.irfft(c, axis=0)[:n]


def orthonormal_dct(y):
    """Computes the 1D Type-II DCT transform such that the transform is orthonormal

    Args:
        y (jax.numpy.ndarray): The 1D real signal

    Returns:
        jax.numpy.ndarray: The orthonormal Type-II Discrete Cosine Transform coefficients of y 

    Orthonormality ensures that

    .. math::

        \\langle a, a \\rangle = \\langle y, y \\rangle
    """
    n = y.shape[0]
    factor = jnp.sqrt(1/(2*n))
    ks = jnp.arange(n)
    phi = jnp.exp(-1j*jnp.pi*ks/(2*n))
    # scaling to make the transform orthonormal
    phi = phi.at[0].set(phi[0]*1/jnp.sqrt(2))
    phi = phi * factor

    y2 = jnp.concatenate( (y[:],  y[::-1]))
    c = jfft.rfft(y2, axis=0)[:n]
    prod = jnp.real(phi*c.T).T
    # phi = phi*jnp.sqrt(2)/n 
    return prod

def orthonormal_idct(a):
    """Computes the 1D Type-II IDCT transform such that the transform is orthonormal

    Args:
        a (jax.numpy.ndarray): The orthonormal Type-II DCT transform coefficients of a 1D real signal

    Returns:
        jax.numpy.ndarray: The 1D real signal y s.t. a = orthonormal_dct(y)
    """
    n = a.shape[0]
    factor = jnp.sqrt(2*n)
    ks = jnp.arange(n)

    phi = jnp.exp(1j*jnp.pi*ks/(2*n))
    # scaling to make the transform orthonormal
    phi = phi*factor
    phi = phi.at[0].set(phi[0]*jnp.sqrt(2))

    upper = (phi*a.T).T
    lower = jnp.zeros((1,)+a.shape[1:])
    c = jnp.concatenate((upper, lower))
    return jfft.irfft(c, axis=0)[:n]
