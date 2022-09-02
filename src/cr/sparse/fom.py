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
First order methods for sparse signal recovery
"""
from cr.sparse._src.fom.util import (
    matrix_affine_func
)

from cr.sparse._src.fom.fom import fom
from cr.sparse._src.fom.scd import (smooth_dual, scd)
from cr.sparse._src.fom.l1rls import (l1rls, l1rls_jit)
from cr.sparse._src.fom.lasso import (lasso, lasso_jit)
from cr.sparse._src.fom.owl1rls import (owl1rls, owl1rls_jit)
from cr.sparse._src.fom.dantzig_scd import (dantzig_scd,)
from cr.sparse._src.fom.bp_scd import (bp_scd,)

from cr.sparse._src.fom.defs import (
    FomOptions,
    FomState
)
