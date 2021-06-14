import pytest
from functools import partial

import jax
from jax import random, jit
import jax.numpy as jnp

import cr.sparse as crs
from cr.sparse import pursuit
import cr.sparse.data as crdata
import cr.sparse.dict as crdict
from cr.sparse.ef import RecoveryPerformance

from cr.sparse._src.pursuit.mp import solve_smv, solve_mmv


solve_smv_jit = jit(solve_smv, static_argnames=("max_iters", "max_res_norm",))
solve_mmv_jit = jit(solve_mmv, static_argnames=("max_iters", "max_res_norm",))

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
dictionary = dictionary.T
# Sparse Representation
x, omega = crdata.sparse_normal_representations(subkey, D, K, S)
representations = x.T
# Signal
signals = representations @ dictionary

def test_mp_smv():
    sol = solve_smv(dictionary, signals[0], max_iters=1, max_res_norm=1)

def test_mp_smv2():
    sol = solve_smv(dictionary, signals[0], max_iters=2, max_res_norm=0)

def test_mp_mmv():
    sol = solve_mmv(dictionary, signals, max_iters=1, max_res_norm=1)

def test_mp_mmv2():
    sol = solve_mmv(dictionary, signals, max_iters=4, max_res_norm=0)

def test_mp_mmv3():
    sol = solve_mmv(dictionary, signals, max_iters=0, max_res_norm=100)
