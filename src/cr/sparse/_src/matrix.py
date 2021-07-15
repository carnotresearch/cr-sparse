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

from .util import promote_arg_dtypes


def transpose(A):
    """Returns the transpose of an array

    Args:
        A (jax.numpy.ndarray): A JAX array

    Returns:
        jax.numpy.ndarray: Transpose of the array
    """
    return jnp.swapaxes(A, -1, -2)

def hermitian(a):
    """Returns the conjugate transpose of an array

    Args:
        A (jax.numpy.ndarray): A JAX array

    Returns:
        jax.numpy.ndarray: Conjugate transpose of the array
    """
    return jnp.conjugate(jnp.swapaxes(a, -1, -2))

def is_matrix(A):
    """Checks if an array is a matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a matrix, False otherwise.
    """
    return A.ndim == 2

def is_square(A):
    """Checks if an array is a square matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a square matrix, False otherwise.
    """
    shape = A.shape
    return A.ndim == 2 and shape[0] == shape[1]

def is_symmetric(A):
    """Checks if an array is a symmetric matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a symmetric matrix, False otherwise.
    """
    shape = A.shape
    if A.ndim != 2: 
        return False
    return jnp.array_equal(A, A.T)

def is_hermitian(A):
    """Checks if an array is a Hermitian matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a Hermitian matrix, False otherwise.
    """
    shape = A.shape
    if A.ndim != 2: 
        return False
    return jnp.array_equal(A, hermitian(A))

def is_positive_definite(A):
    """Checks if an array is a symmetric positive definite matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a symmetric positive definite matrix, False otherwise.

    Symmetric positive definite matrices have real and positive eigen values.
    This function checks if all the eigen values are positive. 
    """
    if not is_symmetric(A):
        return False
    A = promote_arg_dtypes(A)
    return jnp.all(jnp.real(jnp.linalg.eigvals(A)) > 0)


def has_orthogonal_columns(A, atol=1e-6):
    """Checks if a matrix has orthogonal columns

    Args:
        A (jax.numpy.ndarray): A JAX real 2D array


    Returns:
        bool: True if the matrix has orthogonal columns, False otherwise.
    """
    G = A.T @ A
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*m*atol)


def has_orthogonal_rows(A, atol=1e-6):
    """Checks if a matrix has orthogonal rows

    Args:
        A (jax.numpy.ndarray): A JAX real 2D array


    Returns:
        bool: True if the matrix has orthogonal rows, False otherwise.
    """
    G = A @ A.T
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*m*atol)

def has_unitary_columns(A):
    """Checks if a matrix has unitary columns

    Args:
        A (jax.numpy.ndarray): A JAX real or complex 2D array


    Returns:
        bool: True if the matrix has unitary columns, False otherwise.
    """
    G = hermitian(A) @ A
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*1e-6)

def has_unitary_rows(A):
    """Checks if a matrix has unitary rows

    Args:
        A (jax.numpy.ndarray): A JAX real or complex 2D array


    Returns:
        bool: True if the matrix has unitary rows, False otherwise.
    """
    G = A @ hermitian(A)
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*1e-6)
