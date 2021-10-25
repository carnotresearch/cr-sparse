from sls_setup import *

def test_ista1():
    m, n = shape = (11,11)
    A = random.normal(crs.KEYS[0], shape)
    Anp = np.array(A)
    T = lop.jit(lop.matrix(A))
    x = jnp.zeros(n)
    x = x.at[jnp.array([3,5,7])].set(jnp.array([1,1,-1]))
    b = T.times(x)
    # estimate the eignal value of T^H T
    G = lop.jit(lop.gram(T))
    sol = sls.power_iterations(G, jnp.ones(G.shape[0]))
    step_size = float(1 / sol.s)
    # threshold_func = lambda i, x : geo.soft_threshold_percentile_jit(x, 30)
    sol = sls.ista_jit(T, b, x0=jnp.zeros_like(b), step_size=step_size)
    assert_allclose(sol.x, x, atol=1e-2, rtol=1e-2)

