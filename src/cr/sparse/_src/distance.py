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


from scipy.spatial.distance import pdist


# def pdist_rw(X, metric):
#     return pdist(X, metric)

# def pdist_cw(X, metric):
#     return pdist(X.T, metric)


def pairwise_sqr_l2_distances_rw(A, B):
    """
    Computes the pairwise distances between points in A and points in B where each point is a row vector
    """
    M = A.shape[0]
    N = B.shape[0]
    # compute squared sums for each row vector
    a_sums = jnp.sum(A*A, axis=1)
    # reshape to Mx1 column vector
    a_sums = jnp.reshape(a_sums, (M, 1))
    # broadcast to MxN matrix
    a_sums = a_sums * jnp.ones((1,N))

    # compute squared sums for each row vector
    b_sums = jnp.sum(B*B, axis=1)
    # broadcast to MxN matrix
    b_sums = b_sums * jnp.ones((M, 1))

    # multiply A (M x p) and B.T (p x N)
    prods = A @ B.T 
    return a_sums + b_sums - 2 * prods

def pairwise_sqr_l2_distances_cw(A, B):
    """
    Computes the pairwise distances between points in A and points in B where each point is a column vector
    """
    M = A.shape[1]
    N = B.shape[1]
    # compute squared sums for each column vector
    a_sums = jnp.sum(A*A, axis=0)
    # reshape to Mx1 column vector
    a_sums = jnp.reshape(a_sums, (M, 1))
    # broadcast to MxN matrix
    a_sums = a_sums * jnp.ones((1,N))

    # compute squared sums for each column vector
    b_sums = jnp.sum(B*B, axis=0)
    # broadcast to MxN matrix
    b_sums = b_sums * jnp.ones((M, 1))

    # multiply A.T (M x p) and B (p x N)
    prods = A.T @ B 
    return a_sums + b_sums - 2 * prods


def pairwise_l2_distances_rw(A, B):
    return jnp.sqrt(pairwise_sqr_l2_distances_rw(A, B))


def pairwise_l2_distances_cw(A, B):
    return jnp.sqrt(pairwise_sqr_l2_distances_cw(A, B))


def pdist_sqr_l2_rw(A):
    M = A.shape[0]
    # compute squared sums for each row vector
    sums = jnp.sum(A*A, axis=1)
    # broadcast to MxM matrix
    sums = sums * jnp.ones((M, 1))

    # multiply A (M x p) and A.T (p x M)
    prods = A @ A.T 
    return 2*(sums - prods)

def pdist_sqr_l2_cw(A):
    M = A.shape[1]
    # compute squared sums for each col vector
    sums = jnp.sum(A*A, axis=0)
    # broadcast to MxM matrix
    sums = sums * jnp.ones((M, 1))
    # multiply A.T (M x p) and A (p x M)
    prods = A.T @ A 
    return 2*(sums - prods)

def pdist_l2_rw(A):
    return jnp.sqrt(pdist_sqr_l2_rw(A))

def pdist_l2_cw(A):
    return jnp.sqrt(pdist_sqr_l2_cw(A))


def pairwise_l1_distances_rw(A, B):
    return jnp.sum(jnp.abs(A[:, None, :] - B[None, :, :]), axis=-1)

def pairwise_l1_distances_cw(A, B):
    return jnp.sum(jnp.abs(A[:, :, None] - B[:, None, :]), axis=0)

def pdist_l1_rw(A):
    return pairwise_l1_distances_rw(A, A)

def pdist_l1_cw(A):
    return pairwise_l1_distances_cw(A, A)


def pairwise_linf_distances_rw(A, B):
    return jnp.max(jnp.abs(A[:, None, :] - B[None, :, :]), axis=-1)

def pairwise_linf_distances_cw(A, B):
    return jnp.max(jnp.abs(A[:, :, None] - B[:, None, :]), axis=0)


def pdist_linf_rw(A):
    return pairwise_linf_distances_rw(A, A)

def pdist_linf_cw(A):
    return pairwise_linf_distances_cw(A, A)

