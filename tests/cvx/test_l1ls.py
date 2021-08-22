from .cvx_setup import *



def test_l1ls():
    sol = l1ls.solve_jit(Phi, y, 1.)
