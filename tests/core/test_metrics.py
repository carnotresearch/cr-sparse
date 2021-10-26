from core_setup import *

from cr.sparse import metrics


n = 16

z = jnp.zeros(n)
o = jnp.ones(n)

def test_mean_squared():
    assert_almost_equal(metrics.mean_squared(z), 0)
    assert_almost_equal(metrics.mean_squared(o), 1)

def test_mean_squared_error():
    assert_almost_equal(metrics.mean_squared_error(z, z), 0)
    assert_almost_equal(metrics.mean_squared_error(z, o), 1)

def test_root_mean_squared():
    assert_almost_equal(metrics.root_mean_squared(z), 0)
    assert_almost_equal(metrics.root_mean_squared(o), 1)

def test_root_mse():
    assert_almost_equal(metrics.root_mse(z, z), 0)
    assert_almost_equal(metrics.root_mse(z, o), 1)


def test_normalized_root_mse():
    assert_almost_equal(metrics.normalized_root_mse(z, z, 'euclidean'), 0)
    assert_almost_equal(metrics.normalized_root_mse(z, z, 'min-max'), 0)
    assert_almost_equal(metrics.normalized_root_mse(z, z, 'mean'), 0)
    assert_almost_equal(metrics.normalized_root_mse(z, z, 'median'), 0)
    with assert_raises(ValueError):
        metrics.normalized_root_mse(z, z, 'abcd')


def test_peak_signal_noise_ratio():
    assert_almost_equal(metrics.peak_signal_noise_ratio(z, z), 156.5356, decimal=2)


@pytest.mark.parametrize("x,expected", [(o, 168.5768), (z, 0)])
def test_signal_noise_ratio(x, expected):
    assert_almost_equal(metrics.signal_noise_ratio(x, x), expected)
