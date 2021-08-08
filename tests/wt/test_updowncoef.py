from .setup import *


def test_upcoef_and_downcoef_1d_only():
    # upcoef and downcoef raise a ValueError if data.ndim > 1d
    for ndim in [2, 3]:
        data = jnp.ones((8, )*ndim)
        assert_raises(ValueError, wt.downcoef, 'a', data, 'haar')
        assert_raises(ValueError, wt.upcoef, 'a', data, 'haar')
