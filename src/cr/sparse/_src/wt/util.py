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

from .wavelet import build_wavelet, DiscreteWavelet

######################################################################################
# Utility functions
######################################################################################

def dwt_max_level(input_len, filter_len):
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
