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

import cr.sparse.opt as opt


from .util import matrix_affine_func
from .focs import focs
from .defs import FOCSOptions

def l1rls(A, b, lambda_, x0, options: FOCSOptions = FOCSOptions()):
    """Solver for l1 regulated least square problem
    """
    f = opt.smooth_quad_matrix
    h = opt.prox_l1(lambda_)
    af = matrix_affine_func(A, -b)
    return focs(f, af, h, x0, options)
