from .opt_setup import *


def test_project_to_ball():
    n = 32
    x = random.normal(keys[0], (n,))
    y = opt.project_to_ball(x)

def test_project_to_box():
    n = 32
    x = random.normal(keys[0], (n,))
    y = opt.project_to_box(x)

def test_project_to_real_upper_limit():
    n = 32
    x = random.normal(keys[0], (n,))
    y = opt.project_to_real_upper_limit(x)

def test_shrinkage():
    n = 32
    x = random.normal(keys[0], (n,))
    y = opt.shrink(x, 0.5)
