from sls_setup import *



def test_lsqr1():
    m, n = 40, 20
    min_val = 0
    max_val = 50
    A = random.randint(crs.KEYS[0], (m, n), min_val, max_val) / max_val
    x = random.randint(crs.KEYS[1], (n, ), min_val, max_val)
    T = lop.jit(lop.matrix(A))
    # b = A x
    b = T.times(x)
    # initial solution
    x0 = jnp.zeros_like(x)
    # solve using LSQR
    sol = sls.lsqr_jit(T, b, x0, max_iters=25)
    assert_allclose(x, sol.x, atol=1e-3, rtol=1e-5)


def test_lsqr2():
    m, n = 40, 20
    A = crdict.random_onb(crs.KEYS[0], m)[:, :n]
    x = random.normal(crs.KEYS[2], (n,))
    T = lop.jit(lop.matrix(A))
    # b = A x
    b = T.times(x)
    # initial solution
    x0 = jnp.zeros_like(x)
    # solve using LSQR
    sol = sls.lsqr_jit(T, b, x0, max_iters=2)
    assert_allclose(x, sol.x, atol=atol, rtol=rtol)

def test_lsqr3():
    m, n = 40, 20
    A = random.normal(crs.KEYS[0], (m,n)) / jnp.sqrt(m)
    x = random.normal(crs.KEYS[2], (n,))
    T = lop.jit(lop.matrix(A))
    # b = A x
    b = T.times(x)
    # initial solution
    x0 = jnp.zeros_like(x)
    # solve using LSQR
    sol = sls.lsqr_jit(T, b, x0, max_iters=25)
    assert_allclose(x, sol.x, atol=atol, rtol=rtol)


def test_lsqr_state():
    n = 4
    x = jnp.zeros(n)
    state = sls.LSQRState(
        x=x, w=x, u=x, v=x, alpha=x, beta=x,
        rho_bar=x, phi_bar=x, z=x, cs2=0, sn2=0,
        D_norm_sqr=0, cum_z_sqr=0, cum_psi_sqr=0,
        A_norm=0, A_cond=0, x_norm=0, r_norm=0,
        atr_norm=0, iterations=0,
    )
    y = str(state)

def test_lsqr_solution():
    n = 4
    x = jnp.zeros(n)
    res = sls.LSQRSolution(
        x=x,
        A_norm=0, A_cond=0, x_norm=0, r_norm=0,
        atr_norm=0, iterations=0,
    )
    y = str(res)