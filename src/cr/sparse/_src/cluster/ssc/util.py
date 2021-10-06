from jax import jit, vmap
import jax.numpy as jnp

@jit
def sparse_to_full_rep(X, I):
    """Combines values and indices arrays to sparse representations
    """
    # number of signals
    n  = X.shape[1]
    mapper = lambda x, i : jnp.zeros(n).at[i].set(x)
    return vmap(mapper, (1,1), 1)(X, I)