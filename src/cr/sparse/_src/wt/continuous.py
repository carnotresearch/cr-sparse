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

from functools import partial

from jax import jit, lax
import jax.numpy as jnp



def morlet(n, a, w=5):
    """
    Returns the n-point continuous Morlet wavelet

    See the definition at https://en.wikipedia.org/wiki/Morlet_wavelet

    w is the center frequency parameter
    a is the scale parameter
    """
    # n = 3, vec = [-1, 0, 1], n=4 vec=[-1.5, -0.5, 0.5, 1.5]
    # in general [ - (n-1)/2 : (n-1)/2]
    x = jnp.arange(0, n) - (n - 1.0) / 2
    # scale t
    x = x / a
    # wavelet 1 / (pi)^{1/4} e^{j w t / a} e^{-t^2/ a^2}
    wavelet = jnp.exp(1j * w * x) * jnp.exp(-0.5 * x**2) * jnp.pi**(-0.25)
    # energy conservation
    output = jnp.sqrt(1/a) * wavelet
    return output

morlet_jit = jit(morlet, static_argnums=(0,1), static_argnames=("w"))

def ricker(n, a):
    """
    Returns the n-point continuous Ricker/Mexican Hat wavelet function

    See the definition at https://en.wikipedia.org/wiki/Ricker_wavelet
    """
    # The normalization term 2 / (sqrt(3 a) pi^{1/4})
    A = 2 / (jnp.sqrt(3 * a) * (jnp.pi**0.25))
    # square the scale a^2
    wsq = a**2
    # n = 3, vec = [-1, 0, 1], n=4 vec=[-1.5, -0.5, 0.5, 1.5]
    # in general [ - (n-1)/2 : (n-1)/2]
    vec = jnp.arange(0, n) - (n - 1.0) / 2
    # t^2
    xsq = vec**2
    # the modulation term (1 - t^2/a^2)
    mod = (1 - xsq / wsq)
    # the gaussian term e^{-t^2/2a^2}
    gauss = jnp.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total

ricker_jit = jit(ricker, static_argnums=(0,1))


def cwt_complex(data, wavelet_func, scales):
    """Computes the continuous wavelet transform
    """
    wavelet = wavelet_func(1, 1.)
    a = len(scales)
    b = len(data)
    output = jnp.empty((a,b), dtype=wavelet.dtype)
    for index, scale in enumerate(scales):
        n = jnp.minimum(10*scale, b)
        # compute the wavelet
        wavelet = wavelet_func(b, scale)
        # keep a max of 10:scale values
        # wavelet = wavelet[:10*scale]
        # conjugate it
        wavelet = jnp.conj(wavelet)
        # reverse it
        wavelet = wavelet[::-1]
        # convolve with data
        coeffs_real = jnp.convolve(data, wavelet.real, mode='same')
        coeffs_imag = jnp.convolve(data, wavelet.imag, mode='same')
        coeffs = lax.complex(coeffs_real, coeffs_imag)
        output = output.at[index].set(coeffs)
    return output


def cwt(data, wavelet_func, scales):
    """Computes the continuous wavelet transform
    """
    wavelet = wavelet_func(1, 1.)
    a = len(scales)
    b = len(data)
    output = jnp.empty((a,b), dtype=wavelet.dtype)
    for index, scale in enumerate(scales):
        n = jnp.minimum(10*scale, b)
        # compute the wavelet
        wavelet = wavelet_func(b, scale)
        # keep a max of 10:scale values
        # wavelet = wavelet[:10*scale]
        # conjugate it
        wavelet = jnp.conj(wavelet)
        # reverse it
        wavelet = wavelet[::-1]
        # convolve with data
        coeffs = jnp.convolve(data, wavelet, mode='same')
        output = output.at[index].set(coeffs)
    return output


cwt_jit = jit(cwt, static_argnums=(1,))
