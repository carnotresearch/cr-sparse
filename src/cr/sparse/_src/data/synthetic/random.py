

import jax.numpy as jnp
from jax import jit
from jax import random 

def sparse_normal_representations(key, D, K, S):
    """
    Generates a block of representation vectors where each vector is
    K-sparse, the non-zero basis indexes are randomly selected
    and shared among all vectors and non-zero values are normally
    distributed. 

    Args:
        D (int): Dimension of the sparse representation space
        K (int): Number of non-zero entries in the sparse signals
        S (int): Number of sparse signals

    Returns:
        result (DeviceArray): Block of sparse representations
        omega (DeviceArray): Locations of Non-Zero entries
    """
    r = jnp.arange(D)
    r = random.permutation(key, r)
    omega = r[:K]
    omega = jnp.sort(omega)
    shape = [K, S]
    values = random.normal(key, shape)
    result = jnp.zeros([D, S])
    result = result.at[omega, :].set(values)
    result = jnp.squeeze(result)
    return result, omega


def sparse_spikes(key, N, K, S=1):
    """
    Generates a block of representation vectors where each vector is
    K-sparse, the non-zero basis indexes are randomly selected
    and shared among all vectors and non-zero values are Rademacher distributed spikes. 
    """
    key, subkey = random.split(key)
    perm = random.permutation(key, N)
    omega = jnp.sort(perm[:K])
    spikes = jnp.sign(random.normal(subkey, (K,S)))
    X = jnp.zeros((N, S))
    X = X.at[omega, :].set(spikes)
    # reduce the dimension if S = 1
    X = jnp.squeeze(X)
    return X, omega

