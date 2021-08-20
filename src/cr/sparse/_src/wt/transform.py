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
Implements different versions of wavelet transform

n=1024, J=10, L=6

L - J = 10 - 6 = 4.

s_10 = s_6 + d_6 + d_7 + d_8 + d_9

j=J-1:-1:L = [9, 8, 7, 6]

n = 256, J=8, L=0

s_8 = s_0 + d_0 + d_1 + d_2 + d_3 + d_4 + d_5 + d_6 + d_7.

s_J = s_L + sum(L <= j < J) d_j.

"""
from functools import partial

from jax import jit, lax, vmap
import jax.numpy as jnp

from cr.sparse import promote_arg_dtypes

from .dyad import *
from .multirate import *
from .wavelet import build_wavelet
from .util import *

######################################################################################
# Single level wavelet decomposition/reconstruction
######################################################################################

@partial(jit, static_argnums=(1,2))
def pad_(data, p, mode):
    if mode == 'symmetric':
        return jnp.pad(data, p, mode='symmetric')
    elif mode == 'reflect':
        return jnp.pad(data, p, mode='reflect')
    elif mode == 'constant':
        return jnp.pad(data, p, mode='edge')
    elif mode == 'zero':
        return jnp.pad(data, p, mode='constant', constant_values=0)
    elif mode == 'periodic':
        return jnp.pad(data, p, mode='wrap')
    elif mode == 'periodization':
        # Promote odd-sized dimensions to even length by duplicating the
        # last value.
        edge_pad_width = (0, data.shape[0] % 2)
        data = jnp.pad(data, edge_pad_width, mode='edge')
        return jnp.pad(data, p//2, mode='wrap')
    else:
        raise ValueError("mode must be one of ['symmetric', 'constant', 'reflect', 'zero', 'periodic', 'periodization']")



@partial(jit, static_argnums=(3,))
def dwt_(data, dec_lo, dec_hi, mode):
    """Computes single level discrete wavelet decomposition
    """
    p = len(dec_lo)
    x_padded = pad_(data, p, mode)

    x_in = x_padded[None, None, :]

    padding = [(1, 1)] if mode == 'periodization' else [(0, 1)]
    strides = (2,)


    lo_in = dec_lo[::-1][None, None, :]
    lo = lax.conv_general_dilated(x_in, lo_in, strides, padding)
    lo = lo[0, 0, slice(None)]

    hi_in = dec_hi[::-1][None, None, :]
    hi = lax.conv_general_dilated(x_in, hi_in, strides, padding)
    hi =  hi[0, 0, slice(None)]

    if mode == 'periodization':
        #return lo, hi
        return lo[1:-1], hi[1:-1] 
    else:
        return lo[1:-1], hi[1:-1]

def dwt(data, wavelet, mode="symmetric", axis=-1):
    """Computes single level discrete wavelet decomposition
    """
    wavelet = ensure_wavelet_(wavelet)
    data = jnp.asarray(data)
    if jnp.iscomplexobj(data):
        car, cdr = dwt(data.real, wavelet, mode)
        cai, cdi = dwt(data.imag, wavelet, mode)
        return lax.complex(car, cai), lax.complex(cdr, cdi)
    axis = check_axis_(axis, data.ndim)
    data, dec_lo, dec_hi = promote_arg_dtypes(data, wavelet.dec_lo, wavelet.dec_hi)
    if data.ndim == 1:
        return dwt_(data, dec_lo, dec_hi, mode)
    else:
        return dwt_axis_(data, dec_lo, dec_hi, axis, mode)

@partial(jit, static_argnums=(4,))
def idwt_(ca, cd, rec_lo, rec_hi, mode):
    """Computes single level discrete wavelet reconstruction
    """
    p = len(rec_lo)
    ca = up_sample(ca, 2)
    cd = up_sample(cd, 2)
    if mode == 'periodization':
        ca = jnp.pad(ca, p//2, mode='wrap')
        cd = jnp.pad(cd, p//2, mode='wrap')
    # Compute the low pass portion of the next level of approximation
    a = jnp.convolve(ca, rec_lo, 'same')
    # Compute the high pass portion of the next level of approximation
    d = jnp.convolve(cd, rec_hi, 'same')
    # Compute the sum
    sum = a + d
    if mode == 'periodization':
        return sum[p//2:-p//2]
    skip = p//2 - 1
    if skip > 0:
        return sum[skip:-skip]
    return sum

@partial(jit, static_argnums=(3,))
def idwt_joined_(w, rec_lo, rec_hi, mode):
    """Computes single level discrete wavelet reconstruction
    """
    n = len(w)
    m = n // 2
    ca = w[:m]
    cd = w[m:]
    x = idwt_(ca, cd, rec_lo, rec_hi, mode)
    return x


def idwt(ca, cd, wavelet, mode="symmetric", axis=-1):
    """Computes single level discrete wavelet reconstruction
    """
    if ca is None and cd is None:
        raise ValueError("Both ca and cd cannot be None")
    # make sure that ca and cd are arrays
    if ca is not None:
        ca = jnp.asarray(ca)
    if cd is not None:
        cd = jnp.asarray(cd)
    if cd is None:
        cd = jnp.zeros_like(ca)
    if ca is None:
        ca = jnp.zeros_like(cd)
    if ca.shape != cd.shape:
        raise Value("ca and cd must have identical shape.")
    wavelet = ensure_wavelet_(wavelet)
    if ca.shape[0] < wavelet.rec_len // 2:
        raise ValueError("Insufficient coefficients for wavelet reconstruction.")
    axis = check_axis_(axis, ca.ndim)
    if jnp.iscomplexobj(ca) or jnp.iscomplexobj(ca):
        car = jnp.real(ca)
        cai = jnp.imag(ca)
        cdr = jnp.real(cd)
        cdi = jnp.imag(cd)
        xr = idwt(car, cdr, wavelet, mode)
        xi = idwt(cai, cdi, wavelet, mode)
        return lax.complex(xr, xi)
    rec_lo = wavelet.rec_lo
    rec_hi = wavelet.rec_hi
    if ca.ndim == 1:
        return idwt_(ca, cd, rec_lo, rec_hi, mode)
    else:
        return idwt_axis_(ca, cd, rec_lo, rec_hi, axis, mode)


######################################################################################
#  Wavelet decomposition/reconstruction for only one set of coefficients
######################################################################################

@partial(jit, static_argnums=(2,))
def downcoef_(data, filter, mode):
    """Partial discrete wavelet decomposition
    """
    p = len(filter)
    x_padded = pad_(data, p, mode)

    x_in = x_padded[None, None, :]

    padding = [(1, 1)] if mode == 'periodization' else [(0, 1)]
    strides = (2,)

    filter = filter[::-1][None, None, :]
    out = lax.conv_general_dilated(x_in, filter, strides, padding)
    out = out[0, 0, slice(None)]

    if mode == 'periodization':
        return out[1:-1] 
    else:
        return out[1:-1]

def downcoef(part, data, wavelet, mode='symmetric', level=1):
    """Partial discrete wavelet decomposition (multi-level)
    """
    if level < 1:
        raise ValueError("Value of level must be greater than 0.")
    if data.ndim > 1:
        raise ValueError("downcoef only supports 1d data.")
    if jnp.iscomplexobj(data):
        real = downcoef(part, data.real, wavelet, mode, level)
        imag = downcoef(part, data.imag, wavelet, mode, level)
        return lax.complex(real, imag)
    wavelet = ensure_wavelet_(wavelet)
    data = promote_arg_dtypes(data)
    filter = part_dec_filter_(part, wavelet)
    # We do averaging for all levels except the last one
    dec_lo = wavelet.dec_lo
    for i in range(level-1):
        data = downcoef_(data, dec_lo, mode)
    # In the last iteration, we apply 'a' or 'd' as needed
    data = downcoef_(data, filter, mode)
    return data


@partial(jit, static_argnums=(2,))
def upcoef_(coeffs, filter, mode):
    """Partial discrete wavelet reconstruction from one part of coefficients
    """
    p = len(filter)
    coeffs = up_sample(coeffs, 2)
    m = len(coeffs)
    if mode == 'periodization':
        coeffs = jnp.pad(coeffs, p//2, mode='wrap')
    sum = jnp.convolve(coeffs, filter, 'full')
    if mode == 'periodization':
        return sum[p-1:-p]
    return sum[:-1]

@partial(jit, static_argnums=(2,3))
def upcoef_a(coeffs, rec_lo, mode, level):
    for i in range(level):
        coeffs = upcoef_(coeffs, rec_lo, mode)
    return coeffs

@partial(jit, static_argnums=(3,4))
def upcoef_d(coeffs, rec_hi, rec_lo, mode, level):
    coeffs = upcoef_(coeffs, rec_hi, mode)
    for i in range(level-1):
        coeffs = upcoef_(coeffs, rec_lo, mode)
    return coeffs

def upcoef(part, coeffs, wavelet, mode='symmetric', level=1, take=0):
    """Partial discrete wavelet reconstruction from one part of coefficients (multi-level)
    """
    if level < 1:
        raise ValueError("Value of level must be greater than 0.")
    if coeffs.ndim > 1:
        raise ValueError("upcoef only supports 1d coeffs.")
    if jnp.iscomplexobj(coeffs):
        real = upcoef(part, coeffs.real, wavelet, mode, level)
        imag = upcoef(part, coeffs.imag, wavelet, mode, level)
        return lax.complex(real, imag)
    wavelet = ensure_wavelet_(wavelet)
    filter = part_rec_filter_(part, wavelet)
    # We do averaging for all levels except the last one
    rec_lo = wavelet.rec_lo
    coeffs = upcoef_(coeffs, filter, mode)
    for i in range(level-1):
        coeffs = upcoef_(coeffs, rec_lo, mode)
    rec_len = wavelet.rec_len
    if take > 0 and take < rec_len:
        left_bound = right_bound = (rec_len-take) // 2
        if (rec_len-take) % 2:
            # right_bound must never be zero for indexing to work
            right_bound = right_bound + 1
        return coeffs[left_bound:-right_bound]
    return coeffs


######################################################################################
#  Single level wavelet decomposition/reconstruction along a given axis
######################################################################################

@partial(jit, static_argnums=(3,4))
def dwt_axis_(data, dec_lo, dec_hi, axis, mode):
    """Applies the DWT along a given axis
    """
    return jnp.apply_along_axis(dwt_, axis, data, dec_lo, dec_hi, mode)

def dwt_axis(data, wavelet, axis, mode="symmetric"):
    """Computes single level wavelet decomposition along a given axis
    """
    wavelet = ensure_wavelet_(wavelet)
    if jnp.iscomplexobj(data):
        car, cdr = dwt_axis(data.real, wavelet, axis, mode)
        cai, cdi = dwt_axis(data.imag, wavelet, axis, mode)
        return lax.complex(car, cai), lax.complex(cdr, cdi)
    data, dec_lo, dec_hi = promote_arg_dtypes(data, wavelet.dec_lo, wavelet.dec_hi)
    return dwt_axis_(data, dec_lo, dec_hi, axis, mode)



@partial(jit, static_argnums=(4,5))
def idwt_axis_(ca, cd, rec_lo, rec_hi, axis, mode):
    """Applies the Inverse DWT along a given axis
    """
    w = jnp.concatenate((ca, cd), axis=axis)
    return jnp.apply_along_axis(idwt_joined_, axis, w, rec_lo, rec_hi, mode)

def idwt_axis(ca, cd, wavelet, axis, mode="symmetric"):
    """Computes single level wavelet reconstruction along a given axis
    """
    wavelet = ensure_wavelet_(wavelet)
    if ca is not None:
        ca = jnp.asarray(ca)
    if cd is not None:
        cd = jnp.asarray(cd)
    if cd is None:
        cd = jnp.zeros_like(ca)
    if ca is None:
        ca = jnp.zeros_like(cd)
    if ca.shape != cd.shape:
        raise Value("ca and cd must have identical shape.")
    if jnp.iscomplexobj(ca) or jnp.iscomplexobj(ca):
        car = jnp.real(ca)
        cai = jnp.imag(ca)
        cdr = jnp.real(cd)
        cdi = jnp.imag(cd)
        xr = idwt_axis(car, cdr, wavelet, axis, mode)
        xi = idwt_axis(cai, cdi, wavelet, axis, mode)
        return lax.complex(xr, xi)
    return idwt_axis_(ca, cd, wavelet.rec_lo, wavelet.rec_hi, axis, mode)

def dwt_column(data, wavelet, mode="symmetric"):
    """Computes single level wavelet decomposition along columns (axis-0)
    """
    return dwt_axis(data, wavelet, 0, mode)

def dwt_row(data, wavelet, mode="symmetric"):
    """Computes single level wavelet decomposition along rows (axis-1)
    """
    return dwt_axis(data, wavelet, 1, mode)

def dwt_tube(data, wavelet, mode="symmetric"):
    """Computes single level wavelet decomposition along tubes (axis-2)
    """
    return dwt_axis(data, wavelet, 2, mode)

def idwt_column(ca, cd, wavelet, mode="symmetric"):
    """Computes single level wavelet reconstruction along columns (axis-0)
    """
    return idwt_axis(ca, cd, wavelet, 0, mode)

def idwt_row(ca, cd, wavelet, mode="symmetric"):
    """Computes single level wavelet reconstruction along rows (axis-1)
    """
    return idwt_axis(ca, cd, wavelet, 1, mode)

def idwt_tube(ca, cd, wavelet, mode="symmetric"):
    """Computes single level wavelet reconstruction along tubes (axis-2)
    """
    return idwt_axis(ca, cd, wavelet, 2, mode)

######################################################################################
#  Single level wavelet decomposition/reconstruction on 2 dimensions
######################################################################################

dwt2_rw_ = vmap(dwt_, in_axes=(0, None, None, None), out_axes=0)

dwt2_cw_ = vmap(dwt_, in_axes=(1, None, None, None), out_axes=1)

def dwt2(image, wavelet, mode="symmetric", axes=(-2, -1)):
    """Computes single level wavelet decomposition for 2D images
    """
    wavelet = ensure_wavelet_(wavelet)
    image = promote_arg_dtypes(image)
    dec_lo = wavelet.dec_lo
    dec_hi = wavelet.dec_hi
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("Expected two dimensions")
    # make sure that axes are positive
    axes = [a + image.ndim if a < 0 else a for a in axes]
    ca, cd = dwt_axis(image, wavelet, axes[0], mode)
    caa, cad = dwt_axis(ca, wavelet, axes[1], mode)
    cda, cdd = dwt_axis(cd, wavelet, axes[1], mode)
    return caa, (cda, cad, cdd)


def idwt2(coeffs, wavelet, mode="symmetric", axes=(-2, -1)):
    """Computes single level wavelet reconstruction for 2D images
    """
    wavelet = ensure_wavelet_(wavelet)
    caa, (cda, cad, cdd) = coeffs
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("Expected two dimensions")
    # make sure that axes are positive
    axes = [a + caa.ndim if a < 0 else a for a in axes]
    ca = idwt_axis(caa, cad, wavelet, axes[1], mode)
    cd = idwt_axis(cda, cdd, wavelet, axes[1], mode)
    image = idwt_axis(ca, cd, wavelet, axes[0], mode)
    return image


######################################################################################
#  Single level wavelet decomposition/reconstruction on n dimensions
######################################################################################


######################################################################################
#  Multi level wavelet decomposition/reconstruction
######################################################################################

def forward_periodized_orthogonal(qmf, x, L=0):
        """Computes the forward wavelet transform of x

        * Uses the periodized version of x 
        * with an orthogonal wavelet basis
        * length of x must be dyadic.

        if L == 0, then we perform full wavelet decomposition
        """
        # Let's get the dyadic length of x and verify that
        # length of x is a power of 2.
        # assert has_dyadic_length(x)
        n = x.shape[0]
        J = dyadic_length_int(x)
        # assert L < J, "L must be smaller than dyadic index of x"
        # Create the storage for wavelet coefficients.
        end = n

        for j in range(J-1, L-1, -1):
            part = x[:end] 
            # Compute the hipass component of x and downsample it.
            hi = hi_pass_down_sample(qmf, part)
            # Compute the low pass downsampled version
            lo = lo_pass_down_sample(qmf, part)
            # Update the wavelet decomposition
            x = x.at[:end].set(jnp.concatenate((lo, hi)))
            end = end // 2
        return x

forward_periodized_orthogonal_jit = jit(forward_periodized_orthogonal, static_argnums=(2,))



def inverse_periodized_orthogonal(qmf, w, L=0):
        """ Computes the inverse wavelet transform of x

        * Uses the periodized version of x 
        * with an orthogonal wavelet basis
        * length of x must be dyadic.
        """
        # Let's get the dyadic length of w 
        n = w.shape[0]
        J = dyadic_length_int(w)
        # initialize x with its coerce approximation
        mid = 2**L
        lo = w[:mid]
        end = mid*2
        for j in range(L, J):
            hi = w[mid:end]
            # Compute the low pass portion of the next level of approximation
            x_low = up_sample_lo_pass(qmf, lo)
            # Compute the high pass portion of the next level of approximation
            x_hi = up_sample_hi_pass(qmf, hi)
            # Compute the next level approximation of x
            lo = x_low + x_hi
            mid = end
            end = mid * 2
        return lo

inverse_periodized_orthogonal_jit = jit(inverse_periodized_orthogonal, static_argnums=(2,))
