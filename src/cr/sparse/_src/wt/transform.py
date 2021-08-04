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

from jax import jit, lax
import jax.numpy as jnp

from .dyad import *

from .multirate import *

######################################################################################
# Single level wavelet decomposition/reconstruction
######################################################################################

@partial(jit, static_argnums=(3,))
def dwt_(data, dec_lo, dec_hi, mode):
    """Computes single level wavelet decomposition
    """
    p = len(dec_lo)
    if mode == 'symmetric':
        x_padded = jnp.pad(data, p, mode='symmetric')
    elif mode == 'reflect':
        x_padded = jnp.pad(data, p, mode='reflect')
    elif mode == 'constant':
        x_padded = jnp.pad(data, p, mode='edge')
    elif mode == 'zero':
        x_padded = jnp.pad(data, p, mode='constant', constant_values=0)
    elif mode == 'periodic':
        x_padded = jnp.pad(data, p, mode='wrap')
    elif mode == 'periodization':
        x_padded = jnp.pad(data, p//2, mode='wrap')
    else:
        raise ValueError("mode must be one of ['symmetric', 'constant', 'reflect', 'zero', 'periodic', 'periodization']")

    x_in = x_padded[None, None, :]

    padding = [(1, 0)] if mode == 'periodization' else [(0, 0)]
    strides = (2,)


    lo_in = dec_lo[::-1][None, None, :]
    lo = lax.conv_general_dilated(x_in, lo_in, strides, padding)
    lo = lo[0, 0, slice(None)]

    hi_in = dec_hi[::-1][None, None, :]
    hi = lax.conv_general_dilated(x_in, hi_in, strides, padding)
    hi =  hi[0, 0, slice(None)]

    if mode == 'periodization':
        return lo[1:], hi[1:] 
    else:
        return lo[1:-1], hi[1:-1]

def dwt(data, wavelet, mode="symmetric"):
    """Computes single level wavelet decomposition
    """
    return dwt_(data, wavelet.dec_lo, wavelet.dec_hi, mode)


@partial(jit, static_argnums=(4,))
def idwt_(ca, cd, rec_lo, rec_hi, mode):
    """Computes single level wavelet reconstruction
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
    return sum[skip:-skip]


def idwt(ca, cd, wavelet, mode="symmetric"):
    """Computes single level wavelet reconstruction
    """
    return idwt_(ca, cd, wavelet.rec_lo, wavelet.rec_hi, mode)


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
