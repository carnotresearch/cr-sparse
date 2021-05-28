import time
import jax.numpy as jnp
from jax import random

from cr.sparse.la import rq

def print_duration(duration):
    if duration > 1:
        print(f"Duration {duration:.2f} sec")
    else:
        duration = duration * 1000
        if duration > 1:
            print(f"Duration {duration:.0f} msec")
        else:
            duration = duration * 1000
            print(f"Duration {duration:.0f} usec")

def time_rq(seed, M, N):
    key = random.PRNGKey(seed)
    A = random.normal(key, [N, M])
    print(A.shape)
    start = time.time()
    R, Q = rq.factor_mgs(A)
    Q = Q.block_until_ready()
    R = R.block_until_ready()
    end = time.time()
    duration = end - start
    print_duration(duration)
    I = jnp.eye(A.shape[0])
    print ("max(abs(Q Q' - I)):", jnp.max(jnp.abs(Q @ Q.T - I)) )
    print("max(abs(A - RQ))", jnp.max(jnp.abs(A - R @ Q)))
    print("")


time_rq(0, 128, 64)
time_rq(1, 128, 64)
time_rq(2, 128, 64)

print("")
time_rq(0, 64, 28)
time_rq(1, 64, 28)
time_rq(2, 64, 28)

print("")
time_rq(0, 256, 128)
time_rq(1, 256, 128)
time_rq(2, 256, 128)
