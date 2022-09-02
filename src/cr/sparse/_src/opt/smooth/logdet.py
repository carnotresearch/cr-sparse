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
from jax.numpy.linalg import det, cholesky, inv
import cr.nimble as cnb

from .smooth import build2


def smooth_logdet(q=1., C=None):
    r"""Log Det function and its gradient :math:`f(X) = -\log( \text{det}( X ) )`
    """
    q = jnp.asarray(q)
    q = cnb.promote_arg_dtypes(q)
    if C is not None:
        C = jnp.asarray(C)
        C = cnb.promote_arg_dtypes(C)

    @jit
    def func(X):
        X = jnp.asarray(X)
        X = cnb.promote_arg_dtypes(X)
        v = -2*q*jnp.sum(jnp.log(jnp.diag(cholesky(X))))
        if C is not None:
            v = v + cnb.arr_rdot(C, X)
        return v

    @jit
    def gradient(X):
        X = jnp.asarray(X)
        X = cnb.promote_arg_dtypes(X)
        g = -q*inv(X)
        if C is not None:
            g = g + C
        return g

    return build2(func, gradient)

