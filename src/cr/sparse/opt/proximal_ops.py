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


from cr.sparse._src.opt.proximal_ops import (
    prox_value_vec
)


from cr.sparse._src.opt.proximal_ops.prox import (
    ProxCapable,
)

from cr.sparse._src.opt.proximal_ops.prox import build as prox_build
from cr.sparse._src.opt.proximal_ops.prox import build as build_from_ind_proj

from cr.sparse._src.opt.proximal_ops.basic import (
    prox_zero
)

from cr.sparse._src.opt.proximal_ops.lpnorms import (
    prox_l1,
    prox_l2,
    prox_l1_pos,
    prox_l1_ball
)

from cr.sparse._src.opt.proximal_ops.prox_sorted_l1 import prox_owl1