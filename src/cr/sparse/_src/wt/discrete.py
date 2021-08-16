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


from .transform import upcoef

def get_keep_length(output_length: int, level: int, filter_length: int):
    lplus = filter_length  - 2
    keep_length = 1
    for i in range(level):
        keep_length = 2*keep_length + lplus
    return keep_length

def fix_output_length(output_length: int, keep_length: int):
    min_len = keep_length + 2
    output_length = output_length if output_length > min_len else min_len
    return output_length

def get_right_extent_length(output_length, keep_length):
    return output_length - keep_length  - 1

def keep(arr, keep_length):
    length = len(arr)
    if keep_length < length:
        left_bound = (length - keep_length) // 2
        return arr[left_bound:left_bound + keep_length]
    return arr

def d_orth_wavefun(wavelet, level: int=8):
    n = math.pow(math.sqrt(2), level)
    p = math.pow(2, level)
    filter_length = wavelet.dec_len
    output_length = int((filter_length-1) * p + 1)
    keep_length = get_keep_length(output_length, level, filter_length)
    output_length = fix_output_length(output_length, keep_length)

    right_extent_length = get_right_extent_length(output_length,
                                                    keep_length)

    # phi, psi, x
    arr = jnp.array([1.])
    print(level)
    z  = jnp.zeros(1)
    return [jnp.concatenate((z,
                            keep(upcoef('a', arr, wavelet, level=level, take=0), keep_length),
                            jnp.zeros(right_extent_length))),
            jnp.concatenate((z,
                            keep(upcoef('d', arr, wavelet, level=level, take=0), keep_length),
                            jnp.zeros(right_extent_length))),
            jnp.linspace(0.0, (output_length-1)/p, output_length)]

