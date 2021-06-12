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


import jax.numpy as jnp
from jax.ops import index, index_add, index_update
from jax.numpy.linalg import norm
from jax import jit

@jit
def factor_mgs(A):
    n, m = A.shape
    # n rows
    # m dimension
    if n > m:
        raise Exception("Number of rows is larger than dimension")
    Q = jnp.empty([n, m])
    R = jnp.zeros([n, n])
    for k in range(0, n):
        # fill the k-th diagonal entry in R
        atom = A[k]
        norm_a = norm(atom)
        R = index_update(R, index[k, k], norm_a)
        # Initialize the k-th vector in Q
        q = atom / norm_a
        Q = index_update(Q, index[k], q)
        # Compute the inner product of new q vector with each of the remaining rows in A
        products = A[k+1:n, :] @ q.T
        # Place in k-th column of R
        R = index_update(R, index[k+1:n, k], products)
        # Subtract the contribution of previous q vector from all remaining rows of A.
        rr = R[k+1:n, k:k+1]
        update =  -rr @ jnp.expand_dims(q, 0)
        A = index_add(A, index[k+1:n], update)
    return R, Q


def update(R, Q, a, k):
    if k > 0:
        # make it a column vector
        b = jnp.expand_dims(a, 1)
        # Compute the projection of a on each of the previous rows in Q
        h = Q[:k, :] @ b
        # Store in the k-th row of R
        R = index_update(R, index[k, :k], jnp.squeeze(h))
        # subtract the projections
        proj = h.T @ Q[:k, :]
        a = a - jnp.squeeze(proj)
    # compute norm
    a_norm = norm(a)
    # save it in the diagonal entry of R
    R = index_update(R, index[k,k], a_norm)
    # place the new normalized vector
    a = jnp.squeeze(a)
    a = a /a_norm
    Q = index_update(Q, index[k, :], a)
    return R, Q

update = jit(update, static_argnums=(3,))


def solve(R, Q, x):
    pass