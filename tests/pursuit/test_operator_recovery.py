import pytest
from functools import partial

import jax
from jax import random
import jax.numpy as jnp

import cr.sparse as crs
from cr.sparse import pursuit
import cr.sparse.lop as lop
import cr.sparse.data as crdata
from cr.sparse.ef import RecoveryPerformance

from cr.sparse.pursuit import omp
from cr.sparse.pursuit import cosamp
from cr.sparse.pursuit import sp
from cr.sparse.pursuit import iht
from cr.sparse.pursuit import htp

# Iterative Hard Thresholding
iht_solve = partial(iht.operator_solve, normalized=False)
iht_solve_jit = partial(iht.operator_solve_jit, normalized=False)
# Normalized Iterative Hard Thresholding
niht_solve = partial(iht.operator_solve, normalized=True)
niht_solve_jit = partial(iht.operator_solve_jit, normalized=True)

# Hard Thresholding Pursuit
htp_solve = partial(htp.operator_solve, normalized=False)
htp_solve_jit = partial(htp.operator_solve_jit, normalized=False)
# Normalized Hard Thresholding Pursuit
nhtp_solve = partial(htp.operator_solve, normalized=True)
nhtp_solve_jit = partial(htp.operator_solve_jit, normalized=True)

K = 4
M = 40
N = 200

key = random.PRNGKey(8)
key, subkey = random.split(key)
Phi = lop.gaussian_dict(key, M,N)
x, omega = crdata.sparse_normal_representations(subkey, N, K, 1)
x = jnp.squeeze(x)
y = Phi.times(x)



def test_cosamp():
    sol = cosamp.operator_solve_jit(Phi, y, K)
    rp = RecoveryPerformance(Phi, y, x, sol=sol)
    assert rp.success


def test_sp():
    sol = sp.operator_solve_jit(Phi, y, K)
    rp = RecoveryPerformance(Phi, y, x, sol=sol)
    assert rp.success


def test_iht():
    sol = iht_solve_jit(Phi, y, K)
    rp = RecoveryPerformance(Phi, y, x, sol=sol)
    assert rp.success

def test_niht():
    sol = niht_solve_jit(Phi, y, K)
    rp = RecoveryPerformance(Phi, y, x, sol=sol)
    assert rp.success

def test_htp():
    sol = htp_solve_jit(Phi, y, K)
    rp = RecoveryPerformance(Phi, y, x, sol=sol)
    assert rp.success

def test_nhtp():
    sol = nhtp_solve_jit(Phi, y, K)
    rp = RecoveryPerformance(Phi, y, x, sol=sol)
    assert rp.success
    rp.print()
