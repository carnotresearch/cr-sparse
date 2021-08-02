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

from jax import jit, lax
import jax.numpy as jnp

from .dyad import *

from .multirate import *

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

