import pytest
from functools import partial

import jax
from jax import random, jit, vmap
import jax.numpy as jnp

import cr.sparse as crs
from cr.sparse import pursuit
import cr.sparse.data as crdata
import cr.sparse.dict as crdict
import cr.sparse.lop as crlop
from cr.sparse.ef import RecoveryPerformance

from cr.sparse._src.pursuit.mp import solve, matrix_solve


matrix_solve_jit = jit(matrix_solve)
solve_jit = jit(solve, static_argnames=("Phi", ))
matrix_solve_mmv = vmap(matrix_solve_jit, (None, 1), 0)
# Signal dimension
N = 10
# Number of atoms
D = 2*N
# Sparsity level
K = 2
# Number of signals
S = 2

key = random.PRNGKey(0)
key, subkey = random.split(key)
# Dictionary
dictionary = crdict.gaussian_mtx(key, N, D)
dictionary = dictionary
# Sparse Representation
x, omega = crdata.sparse_normal_representations(subkey, D, K, S)
representations = x
# Signal
signals = dictionary @ representations

dict_op = crlop.matrix(dictionary)

def test_mp_smv():
    sol = matrix_solve_jit(dictionary, signals[:, 0], max_iters=1, res_norm_rtol=1e-4)

def test_mp_smv2():
    sol = matrix_solve_jit(dictionary, signals[:, 1], max_iters=2, res_norm_rtol=1e-2)

def test_mp_op_solve():
    sol = solve_jit(dict_op, signals[:, 1], max_iters=1, res_norm_rtol=1e-2)

def test_mp_mmv2():
    sol = matrix_solve_mmv(dictionary, signals)

def test_mp_mmv3():
    Phi = dict_op
    solve_mmv = vmap(lambda x: solve_jit(Phi, x, max_iters=1, res_norm_rtol=1e-2), (1,), 0)
    sol = solve_mmv(signals)
