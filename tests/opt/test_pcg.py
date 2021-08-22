from .opt_setup import *

from cr.sparse.opt import pcg

def test_cg():
    A = jnp.array([[3., 2], [2, 6]])
    b = jnp.array([2., -8])
    op = lambda x : A @ x
    sol = pcg.solve_jit(op, b)
    r = A @ sol.x - b
    assert jnp.isclose(0, r @ r)

