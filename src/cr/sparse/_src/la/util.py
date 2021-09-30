
import jax.numpy as jnp

def hermitian(A):
    r"""Returns the Hermitian transpose of a matrix

    Args:
        A (jax.numpy.ndarray): A matrix

    Returns:
        (jax.numpy.ndarray): A matrix: :math:`A^H`
    """
    return jnp.conjugate(A.T)

def AH_v(A, v):
    r"""Returns :math:`A^H v` for a given matrix A and a vector v

    Args:
        A (jax.numpy.ndarray): A matrix
        v (jax.numpy.ndarray): A vector

    Returns:
        (jax.numpy.ndarray): A vector: :math:`A^H v`

    This is definitely faster on large matrices
    """
    return jnp.conjugate((jnp.conjugate(v.T) @ A).T)