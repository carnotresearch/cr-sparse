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

from jax import jit, lax
import jax.numpy as jnp

import cr.sparse as crs

from .conv import iconv, aconv, mirror_filter


def up_sample(x, s):
    """Upsample x by a factor s by introducing zeros in between
    """
    n = x.shape[0]
    y = jnp.zeros(s*n, dtype=x.dtype)
    return y.at[::s].set(x)

up_sample_jit = jit(up_sample, static_argnums=(1,))

@jit
def lo_pass_down_sample(h, x):
        """Performs low pass filtering followed by downsampling on periodic extension of x

        Reverse the filter and convolve with periodic extension
        """
        # Perform filtering
        y = aconv(h, x)
        # Perform downsampling
        return y[::2]

@jit
def hi_pass_down_sample(h, x):
        """Performs high pass filtering followed by downsampling on periodic extension of x

        Mirror the filter and convolve with periodic extension
        """
        # Construct  the high pass mirror filter 
        g = mirror_filter(h)
        # circular left shift the contents of x by 1.
        x  = crs.vec_rotate_left(x)
        # Perform filtering
        y = iconv(g, x)
        # Perform downsampling
        return y[::2]


@jit
def up_sample_lo_pass(h, x):
        """Performs upsampling followed by low pass filtering
        
        Convolve the filter with periodic extension
        """
        # Upsample by a factor of 2 and introduce zeros
        x = up_sample(x, 2)
        # Perform low pass filtering
        return iconv(h, x)


@jit
def up_sample_hi_pass(h, x):
        """Performs upsampling followed by high pass filtering
        
        Mirror the filter, reverse it and convolve with periodic extension
        """
        # Construct  the high pass mirror filter 
        g = mirror_filter(h)
        # Upsample by a factor of 2 and introduce zeros
        x = up_sample(x, 2)
        # circular right shift the contents of x by 1.
        x  = crs.vec_rotate_right(x)
        # Perform low pass filtering
        return aconv(g, x)


# def up_sample_cdjv(x, h, left_edge, right_edge):
#         """Performs upsampling with filtering and boundary correction"""
#         #TODO complete this one 
#         n = x.shape[0]
#         h_len = h.shape[0]
#         m = h_len // 2
#         # Create a padded version of y
#         y_padded = jnp.zeros(2*n + 3*m + 1, dtype=x.dtype)
#         # fill the middle part with data from x with zero filling
#         # copy n - 2 * m values.
#         start = m+1
#         end = m + 2 * (n  - 2* m)
#         y_padded = y_padded.at[start:end:2].set(x[m: n - m])        
#         # filter
#         y_padded = jnp.convolve(y_padded, h)
#         # Identify left and right edge values
#         left_data = x[:m]
#         right_data = x[n-1:(n - (m  - 1)):-1]
#         # Computed the left and right boundary corrected values
#         left_bc = jnp.vdot(left_edge, left_data)
#         right_bc = jnp.vdot(right_edge, right_data)
#         # final computation of y
#         y = jnp.zeros(2*n, dtype=x.dtype)
#         # copy left boundary corrected values
#         # y(1:3*m - 1) = left_bc(:)
#         # y(2*n:-1:(2*n - 3*m + 2)) = right_bc(:)
#         # add the middle values
#         # y = y + y_padded(1:2*n)
#         return y


def downsampling_convolution_periodization(h, x):
        p = h.shape[0]
        x_padded = jnp.pad(x, p//2, mode='wrap')
        x_in = x_padded[None, None, :]
        y_in = h[::-1][None, None, :]
        out = lax.conv_general_dilated(x_in, y_in, (2,), [(1,0)])
        out = out[0, 0, slice(None)]
        return out[1:]