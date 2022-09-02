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
CR-Sparse
"""
# pylint: disable=W0611


from .version import __version__

# make all the high level cr.nimble api available
from cr.nimble import *

from cr.sparse._src.types import (
    RecoveryFullSolution
)


# Evaluation Tools

from cr.sparse._src.tools.performance import (
    RecoveryPerformance
)

from cr.sparse._src.tools.trials_at_fixed_m_n import (
    RecoveryTrialsAtFixed_M_N
)
