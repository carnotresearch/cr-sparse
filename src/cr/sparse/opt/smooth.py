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


from cr.sparse._src.opt.smooth import (
    smooth_value_grad
)
from cr.sparse._src.opt.smooth.smooth import (
    SmoothFunction,
    smooth_func_translate
)

from cr.sparse._src.opt.smooth.smooth import build as smooth_build
from cr.sparse._src.opt.smooth.smooth import build2 as smooth_build2
from cr.sparse._src.opt.smooth.smooth import build3 as smooth_build3
from cr.sparse._src.opt.smooth.smooth import build_grad_val_func


from cr.sparse._src.opt.smooth.constant import smooth_constant
from cr.sparse._src.opt.smooth.entropy import (
    smooth_entropy,
    smooth_entropy_vg)
from cr.sparse._src.opt.smooth.huber import smooth_huber
from cr.sparse._src.opt.smooth.linear import smooth_linear
from cr.sparse._src.opt.smooth.logdet import smooth_logdet
from cr.sparse._src.opt.smooth.quad import (
    smooth_quad_matrix,
    smooth_quad_error
)
