import time
import jax.numpy as jnp
from jax import random

from cr.sparse.la import rq


def time_rq_update(seed, m, n):
    key = random.PRNGKey(seed)
    A = random.normal(key, [n, m])
    print(A.shape)
    start = time.time()
    Q = jnp.empty([n, m], dtype=jnp.float32)
    R = jnp.zeros([n, n], dtype=jnp.float32)
    for i in range(n):
        R, Q = rq.update(R, Q, A[i], i)
    end = time.time()
    duration = end - start
    print(f"Duration {duration:.1e} sec")
    I = jnp.eye(n)
    print ("max(abs(Q Q' - I)):", jnp.max(jnp.abs(Q @ Q.T - I)) )
    print("max(abs(A - RQ))", jnp.max(jnp.abs(A - R @ Q)))
    print("")

time_rq_update(0, 64, 28)
time_rq_update(1, 64, 28)
time_rq_update(2, 64, 28)
print("")
