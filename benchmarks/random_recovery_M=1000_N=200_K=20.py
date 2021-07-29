from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import cr.sparse as crs
from cr.sparse import pursuit
import cr.sparse.data as crdata
import cr.sparse.dict as crdict
from cr.sparse.ef import RecoveryPerformance

from cr.sparse.pursuit import omp
from cr.sparse.pursuit import cosamp
from cr.sparse.pursuit import sp
from cr.sparse.pursuit import iht
from cr.sparse.pursuit import htp


# Iterative Hard Thresholding
iht_solve_jit = partial(iht.matrix_solve_jit, normalized=False)
# Normalized Iterative Hard Thresholding
niht_solve_jit = partial(iht.matrix_solve_jit, normalized=True)

# Hard Thresholding Pursuit
htp_solve_jit = partial(htp.matrix_solve_jit, normalized=False)
# Normalized Hard Thresholding Pursuit
nhtp_solve_jit = partial(htp.matrix_solve_jit, normalized=True)

K = 20
M = 200
N = 1000

key = random.PRNGKey(8)
keys = random.split(key, 2)
Phi = crdict.gaussian_mtx(keys[0], M,N)
x, omega = crdata.sparse_normal_representations(keys[1], N, K, 1)

y = Phi @ x

def wrap_solve(solver):
    solution = solver(Phi, y, K)
    solution.x_I.block_until_ready()
    solution.r.block_until_ready()
    solution.I.block_until_ready()
    solution.r_norm_sqr.block_until_ready()
    return solution


def time_omp():
    wrap_solve(omp.matrix_solve_jit)


def time_cosamp():
    wrap_solve(cosamp.matrix_solve_jit)

def time_sp():
    wrap_solve(sp.matrix_solve_jit)

def time_iht():
    wrap_solve(iht_solve_jit)

def time_niht():
    wrap_solve(niht_solve_jit)

def time_htp():
    wrap_solve(htp_solve_jit)

def time_nhtp():
    wrap_solve(nhtp_solve_jit)
