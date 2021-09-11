from functools import partial


import jax.numpy as jnp
import numpy as np
from jax import random, jit, grad
import scipy

import cr.sparse as crs
import cr.sparse.la as crla
from cr.sparse import pursuit
import cr.sparse.data as crdata
import cr.sparse.dict as crdict

M = 128
N = 256
K = 8

key = random.PRNGKey(0)
Phi = crdict.gaussian_mtx(key, M,N)

x, omega = crdata.sparse_normal_representations(key, N, K, 1)
x = jnp.squeeze(x)


print(crs.nonzero_indices(x))
print(crs.nonzero_values(x))

y = Phi @ x

from cr.sparse.pursuit import cosamp

solution =  cosamp.solve(Phi, y, K)

print(solution.x_I)
print(solution.I)
print(f"r_norm_sqr: {solution.r_norm_sqr:.2e}, iterations: {solution.iterations}")
print("\n\n")

cosamp_solve  = jit(cosamp.solve, 
    static_argnums=(2), 
    static_argnames=("max_iters", "res_norm_rtol"))

sol = cosamp_solve(Phi, y, K)
print(sol.x_I)
print(sol.I)
print(f"r_norm_sqr: {sol.r_norm_sqr:.2e}, iterations: {sol.iterations}")
