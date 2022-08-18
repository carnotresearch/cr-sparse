from fom_setup import *


def test_1():
    n = 4
    A = jnp.eye(n)
    x = jnp.ones(n)
    b = A @ x
    x0 = jnp.zeros_like(x)
    T = lop.matrix(A)
    sol = l1rls_jit(T, b, 1e-6, x0)
    assert_allclose(sol.x, x, atol=atol, rtol=rtol)