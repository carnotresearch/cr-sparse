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

from jax import jit

import cr.sparse.opt as opt


from .util import matrix_affine_func
from .focs import focs
from .defs import FOCSOptions

def l1rls(A, b, lambda_, x0, options: FOCSOptions = FOCSOptions()):
    r"""Solver for l1 regulated least square problem

    .. math::

        \text{minimize} \frac{1}{2} \| A x - b \|_2^2 + \lambda \| x \|_1 

    """
    f = opt.smooth_quad_matrix()
    h = opt.prox_l1(lambda_)
    return focs(f, h, A, -b, x0, options)


l1rls_jit = jit(l1rls, static_argnums=(0, 4))