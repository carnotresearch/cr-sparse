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


from jax import jit, lax

import jax.numpy as jnp
import cr.nimble as cnb
import cr.sparse.opt as opt

from .prox import build, build_from_ind_proj


def prox_l2(q=1.):
    r"""Returns a prox-capable wrapper for the function :math:`f(x) = \| q x \|_2`

    Returns:
       ProxCapable: A prox-capable function 

    """
    q = jnp.asarray(q)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        v = cnb.arr_l2norm(x)
        return q*v

    @jit
    def proximal_op(x, t):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        v = cnb.arr_l2norm(x)
        s = 1 - 1 / jnp.maximum( v / ( t * q ), 1. )
        x = x * s
        return x

    return build(func, proximal_op)



def prox_l1(q=1.):
    r"""Returns a prox-capable wrapper for the function :math:`f(x) = \| q x \|_1`

    Returns:
       ProxCapable: A prox-capable function 

    """
    q = jnp.asarray(q)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        v = cnb.arr_l1norm(q*x)
        return v

    @jit
    def proximal_op(x, t):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        tq = t * q
        # shrinkage coefficients
        s  = 1 - jnp.minimum( tq/jnp.abs(x), 1 )
        # shrink x
        return x * s

    return build(func, proximal_op)

def prox_l1_pos(q=1.):
    r"""Returns a prox-capable wrapper for the function :math:`f(x) = \| q x \|_1 + I({x \geq 0})`
    
    Returns:
       ProxCapable: A prox-capable function 

    The domain of :math:`f` is restricted to non-negative vectors. This is
    enforced by the indicator function component :math:`I({x \geq 0})` 
    in the definition of :math:`f`.
    """
    q = jnp.asarray(q)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # check if any of the entries in x is negative
        is_invalid = jnp.any(x < 0)
        return lax.cond(is_invalid, 
            # this x is outside the domain
            lambda _: jnp.inf, 
            # x is inside the domain, we compute its l1-norm
            lambda _: cnb.arr_l1norm(q*x),
            None)

    @jit
    def proximal_op(x, t):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        tq = t * q
        # shrinkage only applies on the positive side. negative values are mapped to 0
        return jnp.maximum(0, x - tq)

    return build(func, proximal_op)

def prox_l1_ball(q=1.):
    """Returns a prox-capable wrapper for the l1-ball :math:`\{ x : \| x \|_1 \leq q \}` indicator

    Returns:
       ProxCapable: A prox-capable function 
    """
    ind = opt.indicator_l1_ball(q=q)
    proj = opt.proj_l1_ball(q=q)
    return build_from_ind_proj(ind, proj)
