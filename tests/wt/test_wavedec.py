import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_, assert_raises

import jax.numpy as jnp
from jax import random


from cr.sparse import *
import cr.sparse.wt as wt

key = random.PRNGKey(0)
keys = random.split(key, 16)


def test_compare_downcoef_coeffs():
    r = random.normal(keys[0], (16,))
    # compare downcoef against wavedec outputs
    for nlevels in [1, 2, 3]:
        for wavelet in wt.wavelist(kind='discrete'):
            wavelet = wt.build_wavelet(wavelet)
            if wavelet is not None:
                max_level = wt.dwt_max_level(r.size, wavelet.dec_len)
                if nlevels <= max_level:
                    a = wt.downcoef('a', r, wavelet, level=nlevels)
                    d = wt.downcoef('d', r, wavelet, level=nlevels)
                    coeffs = wt.wavedec(r, wavelet, level=nlevels)
                    assert_allclose(a, coeffs[0])
                    assert_allclose(d, coeffs[1])

