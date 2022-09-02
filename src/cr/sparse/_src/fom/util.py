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

import jax.numpy as jnp
import cr.nimble as cnb
import cr.sparse as crs
from cr.nimble import AH_v

def matrix_affine_func(A=None, b=None):
    """Returns an affine function for a matrix A and vector b
    """
    if A is not None:
        A = jnp.asarray(A)
        A = cnb.promote_arg_dtypes(A)

    if b is not None:
        b = jnp.asarray(b)
        b = cnb.promote_arg_dtypes(b)

    @jit
    def identity(x):
        # both A and b are unspecified. 
        return x

    @jit
    def translate(x):
        # only b is specified.
        return x + b

    @jit
    def similar(x):
        # only A is specified
        return A @ x

    @jit
    def affine(x):
        # both A and b are specified
        return A @ x + b

    ax_plus_b = identity
    if A is None:
        # We assume that A is identity
        ax_plus_b  = translate
    elif b is None:
        # we compute y = A @ x
        ax_plus_b = similar
    else:
        # we compute A @ x + b
        ax_plus_b = affine


    @partial(jit, static_argnums=(2,))
    def operator(x, mode=0):
        if mode == 0:
            return A @ x
        if mode == 1:
            return AH_v(A, x)
        if mode == 2:
            return ax_plus_b(x)

    return operator