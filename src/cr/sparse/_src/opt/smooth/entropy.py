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


from jax import jit, grad, lax

import jax.numpy as jnp
import cr.nimble as cnb

from .smooth import build2, build3

def smooth_entropy():
    r"""Entropy function :math:`f(x) = -\sum(x_i \log (x_i))`
    and its gradient
    """

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        v = lax.cond(jnp.any(x < 0),
            lambda _: -jnp.inf,
            lambda _: - jnp.vdot(x, cnb.log_pos(x)),
            None)
        return v

    @jit
    def gradient(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        g = lax.cond(jnp.any(x < 0),
            lambda _: jnp.full_like(x, jnp.nan),
            lambda _: - cnb.log_pos(x) - 1,
            None)
        return g

    return build2(func, gradient)


def smooth_entropy_vg():
    r"""Entropy function :math:`f(x) = -\sum(x_i \log (x_i))` 
    and its gradient optimized implementation
    """

    def out_of_domain(x):
        v = -jnp.inf
        g = jnp.full_like(x, jnp.nan)
        return g, v

    def in_domain(x):
        logx = cnb.log_pos(x)
        g = -logx - 1
        v = - jnp.vdot(x, logx)
        return g, v


    @jit
    def grad_val(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        g, v = lax.cond(jnp.any(x < 0),
            lambda x: out_of_domain(x),
            lambda x: in_domain(x),
            x)
        return g, v

    basic = smooth_entropy()

    return build3(basic.func, basic.grad, grad_val)
       