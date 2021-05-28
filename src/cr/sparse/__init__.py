# Copyright 2021 Carnot Research Pvt Ltd
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


from cr.sparse._src.util import (
    promote_arg_dtypes
)

from cr.sparse._src.matrix import (
    transpose,
    hermitian,
    is_matrix,
    is_square,
    is_symmetric,
    is_hermitian,
    is_positive_definite,
    has_orthogonal_columns,
    has_orthogonal_rows,
    has_unitary_columns,
    has_unitary_rows,
)

from cr.sparse._src.norm import (
    norms_l1_cw,
    norms_l1_rw,
    norms_l2_cw,
    norms_l2_rw,
    norms_linf_cw,
    norms_linf_rw,

    normalize_l1_cw,
    normalize_l1_rw,
    normalize_l2_cw,
    normalize_l2_rw,
)
