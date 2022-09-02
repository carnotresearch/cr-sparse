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
Algorithms for solving sparse linear systems
"""

# pylint: disable=W0611

from cr.sparse._src.sls.defs import (
    identity_func,
    identity_op,
    default_threshold
)

from cr.sparse._src.sls.lsqr import (
    LSQRSolution,
    LSQRState,
    lsqr,
    lsqr_jit
)

from cr.sparse._src.sls.power import (
    PowerIterSolution,
    power_iterations,
    power_iterations_jit,
)

from cr.sparse._src.sls.ista import (
    ISTAState,
    ista,
    ista_jit,
)

from cr.sparse._src.sls.fista import (
    FISTAState,
    fista,
    fista_jit,
)
