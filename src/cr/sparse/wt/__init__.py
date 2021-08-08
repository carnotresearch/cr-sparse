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

from cr.sparse._src.wt.conv import (
    iconv,
    aconv,
    mirror_filter,
)

from cr.sparse._src.wt.multirate import (
    up_sample,
    lo_pass_down_sample,
    hi_pass_down_sample,
    up_sample_lo_pass,
    up_sample_hi_pass,
    downsampling_convolution_periodization,
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
    ## multi-level transforms
    forward_periodized_orthogonal,
    forward_periodized_orthogonal_jit,
    inverse_periodized_orthogonal,
    inverse_periodized_orthogonal_jit
)

from cr.sparse._src.wt.orth import (
    wavelet_function,
    scaling_function,
    haar,
    db4,
    db6,
    db8,
    db10,
    db12,
    db14,
    db16,
    db18,
    db20,
    baylkin,
    coif1,
    coif2,
    coif3,
    coif4,
    coif5,
    symm4,
    symm5,
    symm6,
    symm7,
    symm8,
    symm9,
    symm10,
    vaidyanathan,
)