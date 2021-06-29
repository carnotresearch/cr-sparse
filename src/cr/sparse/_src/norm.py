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

"""

References

- https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
- https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.norm.html

"""

import jax.numpy as jnp

EPS = jnp.finfo(jnp.float32).eps

norm = jnp.linalg.norm

def norm_l1(x):
    """
    Computes the l_1 norm of a vector
    """
    return jnp.sum(jnp.abs(x))

def sqr_norm_l2(x):
    """
    Computes the squared l_2 norm of a vector
    """
    return x.T @ x

def norm_l2(x):
    """
    Computes the l_2 norm of a vector
    """
    return jnp.sqrt(x.T @ x)

def norm_linf(x):
    """
    Computes the l_inf norm of a vector
    """
    return jnp.max(jnp.abs(x))

def norms_l1_cw(X):
    """
    Computes the l_1 norm of each column of a matrix
    """
    return norm(X, ord=1, axis=0)

def norms_l1_rw(X):
    """
    Computes the l_1 norm of each row of a matrix
    """
    return norm(X, ord=1, axis=1)

def norms_l2_cw(X):
    """
    Computes the l_2 norm of each column of a matrix
    """
    return norm(X, ord=2, axis=0, keepdims=False)

def norms_l2_rw(X):
    """
    Computes the l_2 norm of each row of a matrix
    """
    return norm(X, ord=2, axis=1, keepdims=False)


def norms_linf_cw(X):
    """
    Computes the l_inf norm of each column of a matrix
    """
    return norm(X, ord=jnp.inf, axis=0)

def norms_linf_rw(X):
    """
    Computes the l_inf norm of each row of a matrix
    """
    return norm(X, ord=jnp.inf, axis=1)


def sqr_norms_l2_cw(X):
    """
    Computes the squared l_2 norm of each column of a matrix
    """
    return jnp.sum(X * X, axis=0)

def sqr_norms_l2_rw(X):
    """
    Computes the l_2 norm of each row of a matrix
    """
    return jnp.sum(X * X, axis=1)


######################################
# Normalization of rows and columns
######################################


def normalize_l1_cw(X):
    """
    Normalize each column of X per l_1-norm
    """
    X2 = jnp.abs(X)
    sums = jnp.sum(X2, axis=0) + EPS
    return jnp.divide(X, sums)

def normalize_l1_rw(X):
    """
    Normalize each row of X per l_1-norm
    """
    X2 = jnp.abs(X)
    sums = jnp.sum(X2, axis=1) + EPS
    # row wise sum should be a column vector
    sums = jnp.expand_dims(sums, axis=-1)
    # now broadcasting works well
    return jnp.divide(X, sums)

def normalize_l2_cw(X):
    """
    Normalize each column of X per l_2-norm
    """
    X2 = jnp.square(X)
    sums = jnp.sum(X2, axis=0) 
    sums = jnp.sqrt(sums)
    return jnp.divide(X, sums)

def normalize_l2_rw(X):
    """
    Normalize each row of X per l_2-norm
    """
    X2 = jnp.square(X)
    sums = jnp.sum(X2, axis=1)
    sums = jnp.sqrt(sums)
    # row wise sum should be a column vector
    sums = jnp.expand_dims(sums, axis=-1)
    # now broadcasting works well
    return jnp.divide(X, sums)

