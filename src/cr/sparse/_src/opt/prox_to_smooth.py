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

from jax import jit

import cr.nimble as cnb

import cr.sparse.opt as opt


def moreau_envelope(f: opt.ProxCapable, mu: float = 1.):
    """Constructs a Moreau envelope of a prox-capable function
    """

    @jit
    def func(x):
        px, pv = f.prox_vec_val(x, mu)
        return pv + (1./2) * cnb.arr_rnorm_sqr(x - px)


    @jit
    def grad(x):
        return (1/mu) * (x - f.prox_op(x, mu))

    @jit
    def grad_val(x):
        px, pv = f.prox_vec_val(x, mu)
        v = pv + (1./2) * cnb.arr_rnorm_sqr(x - px)
        g = (1/mu) * (x - px)
        return g, v

    return opt.smooth_build3(func, grad, grad_val)
