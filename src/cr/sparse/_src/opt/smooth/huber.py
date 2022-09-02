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

from .smooth import build2

def smooth_huber(tau=1.):
    r"""Huber penalty function and its gradient
    """
    tau = jnp.asarray(tau)
    tau = cnb.promote_arg_dtypes(tau)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        x_mag = jnp.abs(x)

        small = x_mag <= tau
        x_small = 0.5*(x_mag**2)/tau
        x_large = x_mag - tau/2

        v = jnp.where(small, x_small, x_large)
        return sum(v)

    @jit
    def gradient(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        g = x/jnp.maximum(tau, jnp.abs(x) )
        return g

    return build2(func, gradient)

