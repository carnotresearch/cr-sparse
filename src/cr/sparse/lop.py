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

from cr.sparse._src.lop.lop import (
    # Data type
    Operator,
    # jax support
    jit,
    # operator algebra
    neg,
    scale,
    add,
    subtract,
    compose,
    transpose,
    hermitian,
    hcat,
    power,
    # operator parts
    column,
    columns
)

# basic operators
from cr.sparse._src.lop.basic import (
    identity,
    matrix,
    diagonal,
    zero,
    flipud,
    sum,
    pad_zeros,
    symmetrize,
    restriction,
)

# utilities
from cr.sparse._src.lop.util import (
    to_matrix,
    to_adjoint_matrix,
    to_complex_matrix
)

# The following operators are technically not linear
from cr.sparse._src.lop.basic import (
    real,
)

from cr.sparse._src.lop.signal1d import (
    fourier_basis_1d,
    dirac_fourier_basis_1d
)

from cr.sparse._src.lop.random import (
    gaussian_dict,
    rademacher_dict,
    random_onb_dict,
    random_orthonormal_rows_dict,
)

from cr.sparse._src.lop.props import (
    upper_frame_bound,
)
