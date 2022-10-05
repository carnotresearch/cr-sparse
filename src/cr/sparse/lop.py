# Copyright 2021 CR-Suite Development Team
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
Linear Operators
"""
# pylint: disable=W0611

from cr.sparse._src.lop.lop import (
    # Data type
    Operator,
    # jax support
    jit,
    # operator parts
    column,
    columns
)

# operator algebra
from cr.sparse._src.lop.lop import (
    neg,
    scale,
    partial_op,
    add,
    subtract,
    compose,
    transpose,
    hermitian,
    adjoint,
    hcat,
    power,
    gram,
    frame,
)
from cr.sparse._src.lop.windowed_op import (
    windowed_op
)

from cr.sparse._src.lop.block_diag import (
    block_diag
)

# basic operators
from cr.sparse._src.lop.identity import (
    identity,
)
from cr.sparse._src.lop.dot import (
    dot
)

from cr.sparse._src.lop.spread import (
spread_with_table
)


from cr.sparse._src.lop.basic import (
    real_matrix,
    matrix,
    sparse_real_matrix,
    scalar_mult,
    diagonal,
    zero,
    flipud,
    sum,
    pad_zeros,
    symmetrize,
    restriction,
    heaviside,
    inv_heaviside,
    cumsum,
    diff,
)
from cr.sparse._src.lop.reshape import (
    reshape,
    arr2vec
)

# utilities
from cr.sparse._src.lop.util import (
    to_matrix,
    to_adjoint_matrix,
    to_complex_matrix,
    dot_test_real,
    dot_test_complex,
    rdot_test_complex
)

# The following operators are technically not linear
from cr.sparse._src.lop.basic import (
    real,
)

# Basic signal processing
from cr.sparse._src.lop.filters import (
    running_average,
    fir_filter,
)

# convolutions
from cr.sparse._src.lop.conv import (
    convolve,
    convolve2D,
    convolveND
)

# Orthogonal bases
from cr.sparse._src.lop.onb import (
    fourier_basis,
    dirac_fourier_basis,
    cosine_basis,
    walsh_hadamard_basis,
)

# Fast Fourier Transform
from cr.sparse._src.lop.fft import (
    fft
)


# wavelet transforms
from cr.sparse._src.lop.dwt import (
    dwt,
    dwt2D,
)


# Derivatives
from cr.sparse._src.lop.calculus import (
    first_derivative,
    second_derivative,
)
from cr.sparse._src.lop.tv import (
    tv,
    tv2D
)

# Special matrices
from cr.sparse._src.lop.special_matrices import (
    circulant,
)

# random dictionaries
from cr.sparse._src.lop.random import (
    gaussian_dict,
    rademacher_dict,
    sparse_binary_dict,
    random_onb_dict,
    random_orthonormal_rows_dict,
)

from cr.sparse._src.lop.props import (
    upper_frame_bound,
)
from cr.sparse._src.lop.normest import (
    normest,
    normest_jit
)


# Undocumented
from cr.sparse._src.lop.random import (
    binary_dict_alg,
)