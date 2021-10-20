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


def cont_wave_fun(wavelet, precision):
    """Computes the wavelet function for a Continuous Wavelet
    """
    func = wavelet.functions.time
    lb = wavelet.lower_bound
    ub = wavelet.upper_bound
    n = 2**precision
    t = jnp.linspace(lb, ub, n)
    psi = func(t)
    return psi, t

cont_wave_fun_jit = jit(cont_wave_fun, static_argnums=(0,1))

def int_wave_fun(wavelet, precision):
    """Computes the integral of the wavelet function for a Continuous Wavelet
    """
    psi, t = cont_wave_fun(wavelet, precision)
    dt = t[1] - t[0]
    int_psi = jnp.cumsum(psi) * dt
    return int_psi, t

int_wave_fun_jit = jit(int_wave_fun, static_argnums=(0,1, 2))

def psi_resample(int_psi, dt, domain, scale):
    j = jnp.arange(scale*domain + 1) / (scale * dt)
    j = j.astype(int)  # floor
    int_psi_scale = int_psi[j][::-1]
    return int_psi_scale

psi_resample_jit = jit(psi_resample, static_argnums=(2,3,4))

def cwt_id_time(data, scales, wavelet, precision, axis):
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
    int_psi, t = int_wave_fun(wavelet, precision)
    if not jnp.isrealobj(int_psi):
        int_psi = jnp.conj(int_psi)
    domain = wavelet.domain
    dt = domain / (len(t) - 1)
    a = len(scales)
    n = data.shape[axis]
    out_shape = (a,) + data.shape
    output = jnp.empty(out_shape, dtype=int_psi.dtype)
    # tool to expand the the wavelet across all the data dimensions
    in_slices = [None for _ in data.shape]
    in_slices[axis] = slice(None)
    in_slices = tuple(in_slices)
    for index, scale in enumerate(scales):
        int_psi_scale = psi_resample(int_psi, dt, domain, scale)
        psi_len = len(int_psi_scale)
        filter = int_psi_scale[in_slices]
        if jnp.isrealobj(int_psi):
            conv = jnp.convolve(data, filter)
        else:
            conv_real = jnp.convolve(data, filter.real)
            conv_imag = jnp.convolve(data, filter.imag)
            conv = lax.complex(conv_real, conv_imag)
        coeffs = - jnp.sqrt(scale) * jnp.diff(conv, axis=axis)
        start = psi_len // 2 -1 
        coeffs = jnp.take(coeffs, indices=jnp.arange(start, start+n), axis=axis)
        output = output.at[index].set(coeffs)
    return output

cwt_id_time_jit = jit(cwt_id_time, static_argnums=(1, 2,3,4))

def cwt_id(data, scales, wavelet, method='conv', axis=-1, precision=10):
    """Computes the CWT of data along a specified axis with a specified wavelet
    """
    wavelet = to_wavelet(wavelet)
    if method == 'conv':
        output = cwt_id_time_jit(data, tuple(scales), wavelet, precision, axis=axis)
    else:
        raise NotImplementedError("The specified method is not supported yet")
    return output
    # frequencies = ( wavelet.center_frequency / sampling_period )/ jnp.asarray(scales)
    # return output, frequencies
