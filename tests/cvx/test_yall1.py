from .cvx_setup import *



def test_yall1_rho():
    sol = yall1.solve(Phi, y, rho=1.)

def test_yall1_delta():
    sol = yall1.solve(Phi, y, delta=1.)

def test_yall1_gamma():
    W = lop.identity(N)
    weights = jnp.ones(N)
    sol = yall1.solve(Phi, y, W=W, weights=weights, gamma=1.)

def test_yall1_rho_nojit():
    sol = yall1.solve(Phi, y, rho=1., jit=False)

def test_yall1_delta_nojit():
    sol = yall1.solve(Phi, y, delta=1., jit=False)

def test_yall1_gamma_nojit():
    W = lop.identity(N)
    weights = jnp.ones(N)
    sol = yall1.solve(Phi, y, W=W, weights=weights, gamma=1., jit=False)

# def test_zero_sol():
#     Atb = Phi.trans(y)
#     atb_max = jnp.max(jnp.abs(Atb))
#     rho = float(atb_max) / 2
#     sol = yall1.solve(Phi, y, rho=1.)

# def test_yall1_rho_nng():
#     sol = yall1.solve(Phi, y, nonneg=True, rho=1.)

# def test_yall1_delta_nng():
#     sol = yall1.solve(Phi, y, nonneg=True, delta=1.)

# def test_yall1_gamma_nng():
#     sol = yall1.solve(Phi, y, nonneg=True, gamma=1.)
