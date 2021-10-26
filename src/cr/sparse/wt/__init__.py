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


######################################################################################
# Common Support for Wavelets
######################################################################################

"""
Wavelet Transforms
"""
# pylint: disable=W0611


# Common Utilities 
from cr.sparse._src.wt.util import (
    # Utility functions
    next_pow_of_2,
)


# Dyadic signal processing
from cr.sparse._src.wt.dyad import (
    dyad,
    is_dyadic,
    dyad_to_index,
    dyadic_length,
    has_dyadic_length,
    cut_dyadic,
)

from cr.sparse._src.wt.multirate import (
    dyadup_in,
    dyadup_out,
    up_sample,
    downsampling_convolution_periodization,
)

######################################################################################
# All Wavelets
######################################################################################

from cr.sparse._src.wt.families import (
    FAMILY,
    is_discrete_wavelet,
    families,
    wavelist,
    wname_to_family_order
)

# Functions/Types for both continuous and discrete wavelets
from cr.sparse._src.wt.wavelet import (
    SYMMETRY,
    BaseWavelet,
    build_wavelet,
    to_wavelet,
)

######################################################################################
# Discrete Wavelets
######################################################################################

# Functions/Types for discrete wavelets
from cr.sparse._src.wt.wavelet import (
    DiscreteWavelet,
    build_discrete_wavelet,
)

from cr.sparse._src.wt.transform import (
    pad_,
    # Single level transforms
    # 1D
    dwt_,
    dwt, 
    idwt_,
    idwt,
    # decomposition/reconstruction only for a part
    downcoef_,
    downcoef,
    upcoef_,
    upcoef,
    # along an exis
    dwt_axis_,
    dwt_axis,
    idwt_axis_,
    idwt_axis,
    dwt_column,
    dwt_row,
    dwt_tube,
    idwt_column,
    idwt_row,
    idwt_tube,
    # 2D 
    dwt2,
    idwt2,
)

from cr.sparse._src.wt.discrete import (
    get_keep_length,
    orth_wavefun,
    dwt_coeff_to_arr,
    dwt2_coeff_to_arr
)

from cr.sparse._src.wt.multilevel import (
    wavedec,
    waverec
)


######################################################################################
# Continuous Wavelets
######################################################################################


# Functions/Types for continuous wavelets
from cr.sparse._src.wt.wavelet import (
    build_continuous_wavelet,
    ContinuousWavelet,
    integrate_wavelet,
    central_frequency,
    scale2frequency,
)

# Utilities for discrete wavelets
from cr.sparse._src.wt.util import (
    # definitions
    modes,
    # functions
    make_even_shape,
    pad,
    dwt_max_level,
    dwt_coeff_len,
)

# Utilities for continuous wavelets
from cr.sparse._src.wt.util import (
    time_points,
    frequency_points,
    scales_from_voices_per_octave,
)

# Wavelet functions in time and frequency domains
from cr.sparse._src.wt.cont_wavelets import (
    ricker,
    morlet,
    cmor,
)

# CWT implementation using Torrence and Compo algorithm
from cr.sparse._src.wt.cwt_tc import (
    cwt_tc_time,
    cwt_tc_time_jit,
    cwt_tc_frequency,
    cwt_tc_frequency_jit,
    cwt_tc,
    find_s0,
    find_optimal_scales,
    analyze
)

# CWT implementation using PYWT algorithm
from cr.sparse._src.wt.cwt_int_diff import (
    cont_wave_fun,
    cont_wave_fun_jit,
    int_wave_fun,
    int_wave_fun_jit,
    psi_resample,
    psi_resample_jit,
    cwt_id_time,
    cwt_id_time_jit,
    cwt_id,
)

# Overall CWT interface
from cr.sparse._src.wt.cwt import (
    cwt
)
