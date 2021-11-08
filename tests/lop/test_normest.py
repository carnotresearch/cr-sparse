from .lop_setup import *
from jax.numpy.linalg import norm


@pytest.mark.parametrize("T", [
    lop.fft(64),
    lop.circulant(64, random.normal(crs.KEYS[2], (20,))),
    lop.diagonal(jnp.arange(10))
])
def test_fft_norm(T):
    A = lop.to_matrix(T)
    norm_ref = norm(A, 2)
    norm_est = lop.normest_jit(T)
    assert_almost_equal(norm_ref, norm_est, decimal=3)