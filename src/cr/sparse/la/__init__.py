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
Linear algebra utility functions
"""

from cr.sparse._src.la.linear import (
    point2d,
    vec2d,
    rotate2d_cw,
    rotate2d_ccw,
    reflect2d,
)

from cr.sparse._src.la.triangular import (
    solve_Lx_b,
    solve_LTx_b,
    solve_Ux_b,
    solve_UTx_b,
    solve_spd_chol
)

from cr.sparse._src.la.householder import (
    householder_vec,
    householder_matrix,
    householder_premultiply,
    householder_postmultiply,
    householder_ffm_jth_v_beta,
    householder_ffm_premultiply,
    householder_ffm_backward_accum,
    householder_ffm_to_wy,
    householder_qr_packed,
    householder_split_qf_r,
    householder_qr,
)

from cr.sparse._src.la.chol import (
    cholesky_update_on_add_column,
    cholesky_build_factor
)

# These functions are not JIT ready
from cr.sparse._src.la.householder import (
    householder_vec_
)