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
Wavelet and scaling functions for discrete wavelets
"""
import jax.numpy as jnp
import math


from .transform import upcoef_a,  upcoef_d

from cr.sparse import vec_centered_jit

def get_keep_length(level: int, filter_length: int):
    lplus = filter_length  - 2
    keep_length = 1
    for i in range(level):
        keep_length = 2*keep_length + lplus
    return keep_length

def orth_wavefun(wavelet, level: int=8):
    # the coefficient array [scalar] which will be processed to generate the wavelet and scaling functions
    arr = jnp.array([math.pow(math.sqrt(2), level)])
    # The upsampling factor
    p = math.pow(2, level)
    # length of the filters
    filter_length = wavelet.dec_len
    # the length of scaling and wavelet functions
    output_length = int((filter_length-1) * p + 1)
    # expected number of coefficients in the output of upcoef
    keep_length = get_keep_length(level, filter_length)
    output_length = max(output_length, keep_length + 2)
    # number on zeros on the right side
    # 1 zero in the left. then keep_length entries then extra zeros on the right
    right_extent_length = output_length - keep_length  - 1
    # phi, psi, x
    # a zero to be inserted at the beginning of the scaling / wavelet functions
    z  = jnp.zeros(1)
    # extra zeros to be inserted at the end of the scaling / wavelet functions
    right = jnp.zeros(right_extent_length)
    # The set of values of x for which the scaling / wavelet function has been evaluated.
    x = jnp.linspace(0.0, (output_length-1)/p, output_length)
    mode = 'symmetric'
    # scaling function
    phi = upcoef_a(arr, wavelet.rec_lo, mode, level)
    phi = vec_centered_jit(phi, keep_length)
    phi = jnp.concatenate((z, phi, right))
    # wavelet function
    psi = upcoef_d(arr, wavelet.rec_hi, wavelet.rec_lo, mode, level)
    psi = vec_centered_jit(psi, keep_length)
    psi = jnp.concatenate((z, psi, right))
    # return the result
    return phi, psi, x


def biorth_wavefun(wavelet, level: int=8):
    # the coefficient array [scalar] which will be processed to generate the wavelet and scaling functions
    val = math.pow(math.sqrt(2), level)
    phi_arr = jnp.array([val])
    psi_val = -val if (wavelet.vanishing_moments_psi % 4) != 1 else val
    psi_arr = jnp.array([psi_val])
    # The upsampling factor
    p = math.pow(2, level)
    # length of the filters
    filter_length = wavelet.dec_len
    # the length of scaling and wavelet functions
    output_length = int((filter_length-1) * p )
    # expected number of coefficients in the output of upcoef
    keep_length = get_keep_length(level, filter_length)
    output_length = max(output_length, keep_length + 2)
    # number on zeros on the right side
    # 1 zero in the left. then keep_length entries then extra zeros on the right
    right_extent_length = output_length - keep_length  - 1
    # phi, psi, x
    # a zero to be inserted at the beginning of the scaling / wavelet functions
    z  = jnp.zeros(1)
    # extra zeros to be inserted at the end of the scaling / wavelet functions
    right = jnp.zeros(right_extent_length)
    # The set of values of x for which the scaling / wavelet function has been evaluated.
    x = jnp.linspace(0.0, (output_length-1)/p, output_length)
    mode = 'symmetric'
    # decomposition scaling function
    dec_lo = wavelet.dec_lo[::-1]
    dec_hi = wavelet.dec_hi[::-1]
    phi_d = upcoef_a(phi_arr, dec_lo, mode, level)
    phi_d = vec_centered_jit(phi_d, keep_length)
    phi_d = jnp.concatenate((z, phi_d, right))
    # decomposition wavelet function
    psi_d = upcoef_d(psi_arr, dec_hi, dec_lo, mode, level)
    psi_d = vec_centered_jit(psi_d, keep_length)
    psi_d = jnp.concatenate((z, psi_d, right))
    # reconstruction scaling function
    phi_r = upcoef_a(phi_arr, wavelet.rec_lo, mode, level)
    phi_r = vec_centered_jit(phi_r, keep_length)
    phi_r = jnp.concatenate((z, phi_r, right))
    # reconstruction wavelet function
    psi_r = upcoef_d(psi_arr, wavelet.rec_hi, wavelet.rec_lo, mode, level)
    psi_r = vec_centered_jit(psi_r, keep_length)
    psi_r = jnp.concatenate((z, psi_r, right))
    # return the result
    return phi_d, psi_d, phi_r, psi_r, x

