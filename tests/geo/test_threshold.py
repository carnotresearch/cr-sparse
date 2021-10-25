from geo_setup import *


z = jnp.zeros((4,4))
tau = 0.5
perc = 20.
o = jnp.ones((4,4))

def test_hard_threshold():
    assert_allclose(z, geo.hard_threshold_jit(z, tau))


def test_soft_threshold():
    assert_allclose(z, geo.soft_threshold_jit(z, tau))
    assert_allclose(z, geo.soft_threshold_jit(z+0j, tau))

def test_half_threshold():
    assert_allclose(z, geo.half_threshold_jit(z, tau))



def test_hard_threshold_percentile():
    assert_allclose(z, geo.hard_threshold_percentile_jit(z, perc))


def test_soft_threshold_percentile():
    assert_allclose(z, geo.soft_threshold_percentile_jit(z, perc))
    assert_allclose(z, geo.soft_threshold_percentile_jit(z+0j, perc))

def test_half_threshold_percentile():
    f = 2/3 * o
    assert_allclose(f, geo.half_threshold_percentile_jit(o, perc))


def test_gamma_to_tau():
    gamma = 1.
    tau = geo.gamma_to_tau_half_threshold(gamma)
    tau = geo.gamma_to_tau_hard_threshold(gamma)