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


from .prox import build, build3

def prox_zero():
    r"""Returns a prox-capable wrapper for the function  :math:`f(x)=0`

    Returns:
       ProxCapable: A prox-capable function


    The function :math:`f(x)=0` is the indicator function for the vector space :math:`\RR^n`.

    The proximal operator

    .. math::

        p_f(x, t) = \text{arg} \min_{z \in \RR^n} f(x) + \frac{1}{2t} \| z - x \|_2^2

    reduces to:

    .. math::

        p_f(x, t) = \text{arg} \min_{z \in \RR^n} \frac{1}{2t} \| z - x \|_2^2 = x

    """

    @jit
    def func(x):
        return 0.

    @jit
    def proximal_op(x, t):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return x

    @jit
    def prox_vec_val(x, t):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return x, 0.

    return build3(func, proximal_op, prox_vec_val)

