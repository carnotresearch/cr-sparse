import jax.numpy as jnp

from cr.sparse.la import rq


def process(A):
    print(A)
    n, m = A.shape
    Q = jnp.empty([n, m], dtype=jnp.float32)
    R = jnp.zeros([n, n], dtype=jnp.float32)
    for i in range(n):
        R, Q = rq.update(R, Q, A[i], i)
    # print ("Q Q'\n", Q @ Q.T)
    I = jnp.eye(n)
    print ("max(abs(Q Q' - I)):", jnp.max(jnp.abs(Q @ Q.T - I)) )
    print("max(abs(A - RQ))", jnp.max(jnp.abs(A - R @ Q)))

A = jnp.array([1, 0, 1, 1, 0, -1, 1, 1, 0], dtype=jnp.float32)
A = A.reshape([3,3])
process(A)
