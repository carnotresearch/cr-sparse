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
    """Computes multilevel 1D discrete wavelet transform

    Args:
        data (jax.numpy.ndarray): Input signal array whose DWT is to be computed 
        wavelet (str or cr.sparse.wt.DiscreteWavelet): The wavelet to be used to compute DWT (by name or object) 
        mode (:obj:`str`, optional): Signal extension mode to be used during DWT computation. Default 'symmetric'. 
            See :ref:`Modes <ref-wt-modes>` for available modes.
        axis (int, optional): The axis along which the vectors from data will be picked for computing DWT. Default -1 (last axis).
        level (int, optional): The number of decomposition levels for which DWT will be computed. If the level is 
            unspecified, then it will be computed automatically based on data length and wavelet decomposition filter length.

    Returns:
        :obj:`list` of :obj:`jax.numpy.ndarray`: [cA_n, cD_n, cD_{n-1}, ..., cD_1] A list of wavelet decomposition coefficients of the data. 
            First entry in the tuple is the approximation coefficients
            array at decomposition level n. Second is the detail coefficients array at level n. Third is the 
            detail coefficients array at level n-1. And so on. The last entry in the tuple is the detail 
            coefficients array at level 1 of the wavelet decomposition.
    """
    wavelet = ensure_wavelet_(wavelet)
    data = jnp.asarray(data)
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


def waverec(coeffs, wavelet, mode='symmetric', axis=-1):
    """Multilevel 1D inverse discrete wavelet transform
    """
    if not isinstance(coeffs, (list, tuple)):
        raise ValueError("Expected sequence of coefficient arrays.")
    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 arrays required).")
    elif len(coeffs) == 1:
        # level 0 transform (just returns the approximation coefficients)
        return jnp.asarray(coeffs[0])
    wavelet = ensure_wavelet_(wavelet)
    a, ds = coeffs[0], coeffs[1:]
    if a is not None:
        a = jnp.asarray(a)
    for d in ds:
        if (a is not None) and (d is not None):
            # sometimes a may have one more coefficient than d
            # then it should be dropped
            if a.shape[axis] == d.shape[axis] + 1:
                a = a[tuple(slice(s) for s in d.shape)]
            if a.shape[axis] != d.shape[axis]:
                raise ValueError("coefficient shape mismatch")
        a = idwt_axis(a, d, wavelet, axis, mode)
    return a
