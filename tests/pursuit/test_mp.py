import pytest
from functools import partial

import jax
from jax import random
import jax.numpy as jnp

import cr.sparse as crs
from cr.sparse import pursuit
import cr.sparse.data as crdata
import cr.sparse.dict as crdict
from cr.sparse.ef import RecoveryPerformance

from cr.sparse._src.pursuit.mp import solve_smv, solve_mmv


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
    sol = solve_smv(dictionary, signals[0])
