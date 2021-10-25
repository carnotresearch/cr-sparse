from geo_setup import *




def test_ricker():
    t = jnp.linspace(0, 1, 500)
    w = geo.ricker(t)
