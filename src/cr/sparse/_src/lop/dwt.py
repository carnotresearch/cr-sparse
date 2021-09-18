# Copyright 2021 CR.Sparse Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Linear Operators based on Wavelet Transforms
"""
from functools import partial

from jax import jit, lax
import jax.numpy as jnp

from cr.sparse import wt
from cr.sparse import promote_arg_dtypes

from .lop import Operator
from .util import apply_along_axis

@partial(jit, static_argnums=(3,))
def wavedec(data, dec_lo, dec_hi, level):
    """Compute multilevel wavelet decomposition
    """
    data = promote_arg_dtypes(data)
    a, result = wt.dwt_(data, dec_lo, dec_hi, 'periodization')
    for i in range(level-1):
        a, d = wt.dwt_(a, dec_lo, dec_hi, 'periodization')
        result = jnp.concatenate((d, result))
    result = jnp.concatenate((a, result))
    return result

@partial(jit, static_argnums=(3,))
def waverec(coefs, rec_lo, rec_hi, level):
    """Compute multilevel wavelet reconstruction
    """
    coefs = promote_arg_dtypes(coefs)
    mid = coefs.shape[0] >> level
    a = coefs[:mid]
    end = mid*2
    for j in range(level):
        d = coefs[mid:end]
        a = wt.idwt_(a, d, rec_lo, rec_hi, 'periodization')
        mid = end
        end = mid * 2
    return a

def dwt(n, wavelet="haar", level=1, axis=0):
    """1D Discrete Wavelet Transform operator
    """
    wavelet = wt.to_wavelet(wavelet)
    dec_lo = wavelet.dec_lo
    dec_hi = wavelet.dec_hi
    rec_lo = wavelet.rec_lo
    rec_hi = wavelet.rec_hi

    # We need to verify that the level is not too high
    max_level = wt.dwt_max_level(n, wavelet.dec_len)
    assert level <= max_level, f"Level too high {level=}, {max_level=}"

    m = wt.next_pow_of_2(n)
    pad = (0, m-n)

    def times1d(x):
        x = jnp.pad(x, pad)
        return wavedec(x, dec_lo, dec_hi, level)

    def trans1d(coefs):
        x = waverec(coefs, rec_lo, rec_hi, level)
        return x[:n]

    times, trans = apply_along_axis(times1d, trans1d, axis)
    return Operator(times=times, trans=trans, shape=(m,n))
