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

import math

from jax import jit, lax, vmap
import jax.numpy as jnp

from .wavelet import build_wavelet, DiscreteWavelet

######################################################################################
# Utility functions
######################################################################################

def dwt_max_level(input_len, filter_len):
    """Returns the maximum level of useful DWT decomposition based on data length and filter length
    """
    if isinstance (filter_len, str):
        filter_len = build_wavelet(filter_len)
        if filter_len is None: 
            raise ValueError("Invalid wavelet")
    if isinstance(filter_len, DiscreteWavelet):
        filter_len = filter_len.dec_len
    if filter_len < 2 or int(filter_len) != filter_len:
        raise ValueError("filter_len must be an integer >= 2")
    if input_len < filter_len - 1:
        return 0
    return int(math.log2(input_len // (filter_len - 1)))

# some modes are not supported yet
modes = ["zero", "constant", "symmetric", "periodic", 
            # "smooth",
             "periodization", "reflect", 
             # "antisymmetric", "antireflect"
        ]

def dwt_coeff_len(data_len, filter_len, mode):
    """Returns the length of wavelet decomposition output based on data length, filter length and mode
    """
    if isinstance (filter_len, str):
        filter_len = build_wavelet(filter_len)
        if filter_len is None: 
            raise ValueError("Invalid wavelet")
    if isinstance(filter_len, DiscreteWavelet):
        filter_len = filter_len.dec_len
    if data_len < 1:
        raise ValueError("Value of data_len must be greater than zero.")
    if filter_len < 1:
        raise ValueError("Value of filter_len must be greater than zero.")
    if mode == 'periodization':
        return (data_len + 1) // 2
    else:
        return (data_len + filter_len - 1) // 2


def pad_smooth(vector, pad_width, iaxis, kwargs):
    # smooth extension to left
    left = vector[pad_width[0]]
    slope_left = (left - vector[pad_width[0] + 1])
    vector = vector.at[:pad_width[0]].set(
        left + jnp.arange(pad_width[0], 0, -1) * slope_left)

    # smooth extension to right
    right = vector[-pad_width[1] - 1]
    slope_right = (right - vector[-pad_width[1] - 2])
    vector = vector.at[-pad_width[1]:].set(
        right + jnp.arange(1, pad_width[1] + 1) * slope_right)
    return vector

def pad_antisymmetric(vector, pad_width, iaxis, kwargs):
    # smooth extension to left
    # implement by flipping portions symmetric padding
    npad_l, npad_r = pad_width
    vsize_nonpad = vector.size - npad_l - npad_r
    # Note: must modify vector in-place
    vector = vector.at[:].set(jnp.pad(vector[pad_width[0]:-pad_width[-1]],
                        pad_width, mode='symmetric'))
    r_edge = npad_l + vsize_nonpad - 1
    l_edge = npad_l
    # width of each reflected segment
    seg_width = vsize_nonpad
    # flip reflected segments on the right of the original signal
    n = 1
    while r_edge <= vector.size:
        segment_slice = slice(r_edge + 1,
                                min(r_edge + 1 + seg_width, vector.size))
        if n % 2:
            vector = vector.at[segment_slice].set(vector[segment_slice]*-1)
        r_edge += seg_width
        n += 1

    # flip reflected segments on the left of the original signal
    n = 1
    while l_edge >= 0:
        segment_slice = slice(max(0, l_edge - seg_width), l_edge)
        if n % 2:
            vector.at[segment_slice].set(vector[segment_slice]*-1)
        l_edge -= seg_width
        n += 1
    return vector

def make_even_shape(data):
    """Makes the data shape to be even in all dimensions by duplicating the last value 
    """
    edge_pad_widths = [(0, data.shape[ax] % 2)
                        for ax in range(data.ndim)]
    data = jnp.pad(data, edge_pad_widths, mode='edge')
    return data


def pad(data, pad_widths, mode):
    """Pads a given 1D signal using a given boundary mode.
    """
    data = jnp.asarray(data)
    pad_widths = jnp.asarray(pad_widths)
    if mode == 'symmetric':
        return jnp.pad(data, pad_widths, mode='symmetric')
    elif mode == 'reflect':
        return jnp.pad(data, pad_widths, mode='reflect')
    elif mode == 'antireflect':
        return jnp.pad(data, pad_widths, mode='reflect', reflect_type="odd")
    elif mode == 'constant':
        return jnp.pad(data, pad_widths, mode='edge')
    elif mode == 'zero':
        return jnp.pad(data, pad_widths, mode='constant', constant_values=0)
    elif mode == 'smooth':
        return jnp.pad(data, pad_widths, pad_smooth)
    # elif mode == 'antisymmetric':
    #     return jnp.pad(data, pad_widths, pad_antisymmetric)
    elif mode == 'periodic':
        return jnp.pad(data, pad_widths, mode='wrap')
    elif mode == 'periodization':
        # Promote odd-sized dimensions to even length by duplicating the
        # last value.
        edge_pad_widths = [(0, data.shape[ax] % 2)
                            for ax in range(data.ndim)]
        data = jnp.pad(data, edge_pad_widths, mode='edge')
        return jnp.pad(data, pad_widths, mode='wrap')
    else:
        raise ValueError("mode must be one of ['symmetric', 'constant', 'reflect', 'antireflect', 'zero', 'smooth', 'periodic',  'periodization']")


######################################################################################
# Local utility functions
######################################################################################

def ensure_wavelet_(wavelet):
    if isinstance(wavelet, str):
        wavelet = build_wavelet(wavelet)
    if wavelet is None:
        raise ValueError("Invalid wavelet")
    return wavelet


def part_dec_filter_(part, wavelet):
    if part == 'a':
        return wavelet.dec_lo
    if part == 'd':
        return wavelet.dec_hi
    raise ValueError(f'Invalid part: {part}')

def part_rec_filter_(part, wavelet):
    if part == 'a':
        return wavelet.rec_lo
    if part == 'd':
        return wavelet.rec_hi
    raise ValueError(f'Invalid part: {part}')


def check_axis_(axis, ndim):
    if axis < 0:
        axis = ndim + axis
    if axis >= ndim:
        raise ValueError(f"Invalid axis: {axis} with ndim: {ndim}")
    return axis