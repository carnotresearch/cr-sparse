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
import cr.sparse as crs


def matrix_affine_func(A=None, b=None):
    """Returns an affine function for a matrix A and vector b
    """
    if A is not None:
        A = jnp.asarray(A)
        A = crs.promote_arg_dtypes(A)

    if b is not None:
        b = jnp.asarray(b)
        b = crs.promote_arg_dtypes(b)

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

    if A is None and b is None:
        return identity

    if A is None:
        # We assume that A is identity
        return translate

    if b is None:
        return similar

    return affine