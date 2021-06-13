import pytest

# jax imports
import jax
from jax import jit
import jax.numpy as jnp

# crs imports
import cr.sparse.la as crla

householder_vec  = jit(crla.householder_vec)
householder_matrix = jit(crla.householder_matrix)
householder_premultiply = jit(crla.householder_premultiply)
householder_postmultiply = jit(crla.householder_postmultiply)
householder_ffm_backward_accum = jit(crla.householder_ffm_backward_accum)
householder_qr = jit(crla.householder_qr)
householder_qr_packed = jit(crla.householder_qr_packed)
householder_split_qf_r = jit(crla.householder_split_qf_r)
householder_ffm_premultiply = jit(crla.householder_ffm_premultiply)
householder_ffm_to_wy = jit(crla.householder_ffm_to_wy)
A = jnp.array([[12.0,-51, 4], [6, 167, -68], [-4, 24, -41]])

def test_vec():
    x = A[:, 0]
    v, beta = householder_vec(x)
    v_expected = jnp.array([ 1., -3.,  2.])
    assert jnp.allclose(v, v_expected)
    assert jnp.isclose(beta, 0.14285715)

def test_vec_():
    x = A[:, 0]
    v, beta = crla.householder_vec_(x)
    v_expected = jnp.array([ 1., -3.,  2.])
    assert jnp.allclose(v, v_expected)
    assert jnp.isclose(beta, 0.14285715)

def test_vec2():
    x = jnp.array([1.0, 0, 0])
    v, beta = crla.householder_vec_(x)

def test_vec3():
    x = jnp.array([0, 1., 0])
    v, beta = crla.householder_vec_(x)

def test_vec4():
    x = jnp.array([-2, 0., 0])
    v, beta = crla.householder_vec_(x)

def test_vec5():
    x = jnp.array([1, 1., 0])
    v, beta = crla.householder_vec_(x)


def test_vec6():
    x = jnp.array([-1, 1., 0])
    v, beta = crla.householder_vec_(x)

def test_vec7():
    x = jnp.array([-1])
    v, beta = crla.householder_vec_(x)

def test_vec8():
    x = jnp.array([-1])
    v, beta = householder_vec(x)

def test_matrix():
    x = A[:, 0]
    Q = jnp.array([[ 6.,  3., -2.],
             [ 3., -2.,  6.],
             [-2.,  6.,  3.]]) / 7
    P = householder_matrix(x)
    assert jnp.allclose(Q, P)


def test_premultiply():
    x = A[:, 0]
    v, beta = householder_vec(x)
    A2  = householder_premultiply(v, beta, A)
    expected = jnp.array([[ 14.,  21., -14.],
             [  0., -49., -14.],
             [  0., 168., -77.]])
    assert jnp.allclose(expected, A2, atol=1e-3)

def test_postmultiply():
    x = A[:, 0]
    v, beta = householder_vec(x)
    A2  = householder_postmultiply(v, beta, A.T)
    expected = jnp.array([[ 14.,  21., -14.],
             [  0., -49., -14.],
             [  0., 168., -77.]])
    assert jnp.allclose(expected.T, A2, atol=1e-3)


def test_qr():
    Q, R = householder_qr(A)
    assert jnp.allclose(A, Q @ R, atol=1e-3)


def test_fftm_to_wy():
    Q, R = householder_qr(A)
    A2 = householder_qr_packed(A)
    W, Y = householder_ffm_to_wy(A2)
    I = jnp.eye(A.shape[0])
    Qb = I - W @ Y.T
    assert jnp.allclose(Q, Qb)

def test_ffm_premultiply():
    Q, R = householder_qr(A)
    A2 = householder_qr_packed(A)
    QF, R = householder_split_qf_r(A2)
    I = jnp.eye(A.shape[0])
    C = householder_ffm_premultiply(QF, I)
