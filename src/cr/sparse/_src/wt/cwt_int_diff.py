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


"""
CWT implementation in terms of integrating the wavelet, resampling it,
convolving it and then differentiating the result.

This technique is followed in pywt.
"""
from functools import partial

from jax import jit, lax
import jax.numpy as jnp
import jax.numpy.fft as jfft

import cr.sparse as crs

from .wavelet import to_wavelet, integrate_wavelet, scale2frequency

def cwt_time_int_diff(data, int_psi, t, scales, axis=-1):
    """Computes the CWT using time domain convolution

    It uses the following method
    - compute the wavelet
    - integrate it
    - resample the wavelet for different scales
    - convolve the wavelet with the data
    - differentiate the result
    - scale the result

    This method is what is followed in PyWavelets.
    """
    data = jnp.asarray(data)
    a = len(scales)
    n = data.shape[axis]
    out_shape = (a,) + data.shape
    output = jnp.empty(out_shape, dtype=int_psi.dtype)
    # tool to expand the the wavelet across all the data dimensions
    in_slices = [None for _ in data.shape]
    in_slices[axis] = slice(None)
    in_slices = tuple(in_slices)
    dt = t[1] - t[0]
    domain = t[-1] - t[0]
    for index, scale in enumerate(scales):
        j = jnp.arange(scale*domain + 1) / (scale * dt)
        j = j.astype(int)  # floor
        int_psi_scale = int_psi[j][::-1]
        psi_len = len(int_psi_scale)
        conv = jnp.convolve(data, int_psi_scale[in_slices])
        coeffs = - jnp.sqrt(scale) * jnp.diff(conv, axis=axis)
        start = psi_len // 2 -1 
        coeffs = jnp.take(coeffs, indices=range(start, start+n), axis=axis)
        output = output.at[index].set(coeffs)
    return output

# cwt_time_int_diff = jit(cwt_time_int_diff, static_argnums=(3,), static_argnames=(4,))

def cwt(data, scales, wavelet, sampling_period=1., axis=-1, precision=10, method='conv'):
    """Computes the CWT of data along a specified axis with a specified wavelet
    """
    wavelet = to_wavelet(wavelet)
    if method == 'conv':
        int_psi, t = integrate_wavelet(wavelet, precision=precision)
        int_psi = jnp.conj(int_psi)
        output = cwt_time_int_diff(data, int_psi, t, scales, axis=axis)
    else:
        raise NotImplementedError("The specified method is not supported yet")
    frequencies = scale2frequency(wavelet, scales, precision)
    frequencies /= sampling_period
    return output, frequencies
