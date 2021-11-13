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

import jax.numpy as jnp
from jax.numpy.linalg import qr, norm

import cr.sparse as crs

def indicator_l2_ball(q=1., b=None, A=None):

    if b is not None:
        b = jnp.asarray(b)
        b = crs.promote_arg_dtypes(b)

    if A is not None:
        A = jnp.asarray(A)
        A = crs.promote_arg_dtypes(A)

    if q <= 0:
        raise ValueError("q must be greater than 0")

    @jit
    def indicator_q(x):
        invalid = norm(x) > q
        return jnp.where(invalid, jnp.inf, 0)

    if b is None and A is None:
        return indicator_q


    @jit
    def indicator_q_b(x):
        # compute difference from center
        r = x - b
        invalid = norm(r) > q
        return jnp.where(invalid, jnp.inf, 0)

    if A is None:
        return indicator_q_b

    if b is None:
        # we have q and A specified. 
        # default value for b
        b = 0.

    @jit
    def indicator_q_b_A(x):
        # compute the residual vector
        r = A @ x - b
        invalid = norm(r) > q
        return jnp.where(invalid, jnp.inf, 0)
    
    return indicator_q_b_A
