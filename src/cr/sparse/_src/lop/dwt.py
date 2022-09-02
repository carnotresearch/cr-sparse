# Copyright 2021 CR-Suite Development Team
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

import numpy as np
from jax import jit, lax
import jax.numpy as jnp

import cr.wavelets as wt
from cr.nimble import promote_arg_dtypes

from .lop import Operator
from .util import apply_along_axis

@partial(jit, static_argnums=(3,))
def wavedec(data, dec_lo, dec_hi, level):
    """Compute multilevel wavelet decomposition
    """
    data, dec_lo, dec_hi = promote_arg_dtypes(data, dec_lo, dec_hi)
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
    coefs, rec_lo, rec_hi = promote_arg_dtypes(coefs, rec_lo, rec_hi)
    mid = coefs.shape[0] >> level
    a = coefs[:mid]
    end = mid*2
    for j in range(level):
        d = coefs[mid:end]
        a = wt.idwt_(a, d, rec_lo, rec_hi, 'periodization')
        mid = end
        end = mid * 2
    return a

@partial(jit, static_argnums=(3,4))
def wavedec2(image, dec_lo, dec_hi, level, axes):
    """Compute multilevel wavelet decomposition of 2D images
    """
    image, dec_lo, dec_hi = promote_arg_dtypes(image, dec_lo, dec_hi)
    ax0 = axes[0]
    ax1 = axes[1]
    mode = 'periodization'
    result = image
    slices = [slice(None) for _ in range(image.ndim)]
    slices[ax0] = slice(0, image.shape[ax0])
    slices[ax1] = slice(0, image.shape[ax1])
    for i in range(level):
        ca, cd = wt.dwt_axis_(image, dec_lo, dec_hi, ax0, mode)
        caa, cad = wt.dwt_axis_(ca, dec_lo, dec_hi, ax1, mode)
        cda, cdd = wt.dwt_axis_(cd, dec_lo, dec_hi, ax1, mode)
        ca = jnp.concatenate((caa, cad), axis=ax1)
        cd = jnp.concatenate((cda, cdd), axis=ax1)
        coefs = jnp.concatenate((ca, cd), axis=ax0)
        result = result.at[tuple(slices)].set(coefs)
        image = caa
        slices[ax0] = slice(0, caa.shape[ax0])
        slices[ax1] = slice(0, caa.shape[ax1])
    return result

@partial(jit, static_argnums=(3,4))
def waverec2(coefs, rec_lo, rec_hi, level, axes):
    """Compute multilevel wavelet reconstruction for 2D images
    """
    coefs = promote_arg_dtypes(coefs)
    ax0 = axes[0]
    ax1 = axes[1]
    mode = 'periodization'
    mid0 = coefs.shape[ax0] >> level
    mid1 = coefs.shape[ax1] >> level
    slices = [slice(None) for _ in range(coefs.ndim)]
    slices[ax0] = slice(0, mid0)
    slices[ax1] = slice(0, mid1)
    caa = coefs[tuple(slices)]
    end0 = mid0*2
    end1 = mid1*2
    for j in range(level):
        # cad
        slices[ax0] = slice(0, mid0)
        slices[ax1] = slice(mid1, end1)
        cad = coefs[tuple(slices)]
        # cda
        slices[ax0] = slice(mid0, end0)
        slices[ax1] = slice(0, mid1)
        cda = coefs[tuple(slices)]
        # cdd
        slices[ax0] = slice(mid0, end0)
        slices[ax1] = slice(mid1, end1)
        cdd = coefs[tuple(slices)]
        # ca
        ca = wt.idwt_axis_(caa, cad, rec_lo, rec_hi, ax1, mode)
        # cd
        cd = wt.idwt_axis_(cda, cdd, rec_lo, rec_hi, ax1, mode)
        # combine ca,cd
        caa = wt.idwt_axis_(ca, cd, rec_lo, rec_hi, ax0, mode)
        # now update the ranges for next round
        mid0 = end0
        mid1 = end1
        end0 = mid0*2
        end1 = mid1*2
    return caa


def dwt(n, wavelet="haar", level=1, axis=0, basis=False):
    """Returns a 1D Discrete Wavelet Transform operator

    Args:
        n (int): Dimension of the input signal and output coefficients  
        wavelet (string): Name of the discrete wavelet to be used
        level (int): Number of wavelet decompositions (default 1) 
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied
        basis (bool): If False, the transform operator is returned. 
          If True, the wavelet basis operator is returned instead. Default False. 

    Returns:
        Operator: A linear operator wrapping 1D DWT transform or basis
    """
    wavelet = wt.to_wavelet(wavelet)
    dec_lo = wavelet.dec_lo
    dec_hi = wavelet.dec_hi
    rec_lo = wavelet.rec_lo
    rec_hi = wavelet.rec_hi

    # We need to verify that the level is not too high
    max_level = wt.dwt_max_level(n, wavelet.dec_len)
    assert level <= max_level, f"Level too high level={level}, max_level={max_level}"

    m = wt.next_pow_of_2(n)
    pad = (0, m-n)

    def times1d(x):
        x = jnp.pad(x, pad)
        return wavedec(x, dec_lo, dec_hi, level)

    def trans1d(coefs):
        x = waverec(coefs, rec_lo, rec_hi, level)
        return x[:n]

    times, trans = apply_along_axis(times1d, trans1d, axis)
    if basis:
        # Return the wavelet basis
        return Operator(times=trans, trans=times, shape=(n,m))
    else:
        # Return the wavelet transform
        return Operator(times=times, trans=trans, shape=(m,n))


def dwt2D(shape, wavelet="haar", level=1, axes=None, basis=False):
    """Returns a 2D Discrete Wavelet Transform operator

    Args:
        shape (tuple): Shape of input image / output coefficients  
        wavelet (string): Name of the discrete wavelet to be used
        level (int): Number of wavelet decompositions (default 1) 
        axes (tuple): For multi-dimensional array input, the pair of axes along which
          the linear operator will be applied
        basis (bool): If False, the transform operator is returned. 
          If True, the wavelet basis operator is returned instead. Default False. 

    Returns:
        Operator: A linear operator wrapping 2D DWT transform or basis
    """
    wavelet = wt.to_wavelet(wavelet)
    dec_lo = wavelet.dec_lo
    dec_hi = wavelet.dec_hi
    rec_lo = wavelet.rec_lo
    rec_hi = wavelet.rec_hi

    # Make sure that input shape is more than 2D
    assert len(shape) >= 2, f"Input shape must be 2 or more dimensional"

    if axes is None:
        # By default, the DWT will happen over the first 2 dimensions
        axes = (0,1)
    else:
        axes = tuple(axes)
    h = shape[axes[0]]
    w = shape[axes[1]]

    # We need to verify that the level is not too high
    max_level_1 = wt.dwt_max_level(h, wavelet.dec_len)
    max_level_2 = wt.dwt_max_level(w, wavelet.dec_len)
    max_level = min(max_level_1, max_level_2)
    assert level <= max_level, f"Level too high level={level}, max_level={max_level}"

    hh = wt.next_pow_of_2(h)
    ww = wt.next_pow_of_2(w)
    pad = ((0, hh-h), (0, ww-w))

    out_shape = list(shape)
    out_shape[axes[0]] = hh
    out_shape[axes[1]] = ww
    out_shape = tuple(out_shape)

    def times(x):
        x = jnp.pad(x, pad)
        return wavedec2(x, dec_lo, dec_hi, level, axes)

    def trans(coefs):
        x = waverec2(coefs, rec_lo, rec_hi, level, axes)
        return x[:h, :w]

    if basis:
        # Return the wavelet basis
        return Operator(times=trans, trans=times, shape=(shape, out_shape))
    else:
        # Return the wavelet transform
        return Operator(times=times, trans=trans, shape=(out_shape, shape))
