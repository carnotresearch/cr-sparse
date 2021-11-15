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
Optimization Utilities
"""
# pylint: disable=W0611


from cr.sparse._src.opt.projections import (
    project_to_ball,
    project_to_box,
    project_to_real_upper_limit
)

from cr.sparse._src.opt.shrinkage import (
    shrink
)


from .indicators import *
from .projectors import *
from .proximal_ops import *
from .smooth import *
