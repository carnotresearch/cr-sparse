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

"""
Multi-level wavelet transforms
"""

from functools import partial

import numpy as np
from jax import jit, lax
import jax.numpy as jnp


from cr.sparse import promote_arg_dtypes

from .util import *
from .transform import *


def check_level_(sizes, dec_lens, level):
    if jnp.isscalar(sizes):
        sizes = (sizes, )
    if jnp.isscalar(dec_lens):
        dec_lens = (dec_lens, )
    max_level = np.min([dwt_max_level(s, d) for s, d in zip(sizes, dec_lens)])
    if level is None:
        level = max_level
    elif level < 0:
        raise ValueError(
            "Level value of %d is too low . Minimum level is 0." % level)
    elif level > max_level:
        warnings.warn(
            ("Level value of {} is too high: all coefficients will experience "
             "boundary effects.").format(level))
    return level


def wavedec(data, wavelet, mode='symmetric', level=None, axis=-1):
    """
    Multilevel 1-D transform
    """
    wavelet = ensure_wavelet_(wavelet)
    data = promote_arg_dtypes(data)
    try:
        axes_shape = data.shape[axis]
    except IndexError:
        raise ValueError("Axis greater than data dimensions")

    level = check_level_(axes_shape, wavelet.dec_len, level)

    coeffs_list = []
    a = data
    for i in range(level):
        a, d = dwt_axis(a, wavelet, axis, mode)
        coeffs_list.append(d)
    coeffs_list.append(a)
    coeffs_list.reverse()
    return coeffs_list
