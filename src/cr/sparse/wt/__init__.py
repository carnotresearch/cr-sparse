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

from cr.sparse._src.wt.families import (
    FAMILY,
    is_discrete_wavelet,
    families,
    wavelist,
    wname_to_family_order
)

from cr.sparse._src.wt.wavelet import (
    SYMMETRY,
    BaseWavelet,
    DiscreteWavelet,
    build_discrete_wavelet,
    build_wavelet,
)


from cr.sparse._src.wt.dyad import (
    dyad,
    is_dyadic,
    dyad_to_index,
    dyadic_length,
    has_dyadic_length,
    cut_dyadic,
)

from cr.sparse._src.wt.util import (
    # utility definitions
    modes,
    # Utility functions
    make_even_shape,
    pad,
    dwt_max_level,
    dwt_coeff_len,
)

from cr.sparse._src.wt.transform import (
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
    orth_wavefun
)

from cr.sparse._src.wt.continuous import (
    next_pow_of_2,
    time_points,
    frequency_points,
    ricker,
    morlet,
    cwt_time_real,
    cwt_time_real_jit,
    cwt_time_complex,
    cwt_time_complex_jit,
    cwt_frequency,
    cwt_frequency_jit,
    find_s0,
    find_optimal_scales,
    analyze
)

from cr.sparse._src.wt.multirate import (
    up_sample,
    downsampling_convolution_periodization,
)


from cr.sparse._src.wt.multilevel import (
    wavedec,
    waverec
)

