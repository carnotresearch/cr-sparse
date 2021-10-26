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
Algorithms and utilities for solving SVD problems
"""

from cr.sparse._src.la.svdpack.reorth import (
    reorth_mgs,
    reorth_mgs_jit,
    reorth_noop
)

from cr.sparse._src.la.svdpack.bdsqr import (
    bdsqr,
    bdsqr_jit
)

from cr.sparse._src.la.svdpack.lanbpro_utils import (
    LanBDOptions,
    LanBProState,
    lanbpro_options_init,
    do_elr,
    lanbpro_random_start,
    update_nu,
    update_mu,
    compute_ind,
    bpro_norm_estimate,
)

from cr.sparse._src.la.svdpack.lanbpro import (
    lanbpro_init,
    lanbpro_iteration,
    lanbpro_iteration_jit,
    lanbpro,
    lanbpro_jit,
    new_r_vec,
    new_p_vec,
)
from cr.sparse._src.la.svdpack.lansvd_utils import (
    refine_bounds,
)

from cr.sparse._src.la.svdpack.lansvd import (
    lansvd_simple,
    lansvd_simple_jit
)
