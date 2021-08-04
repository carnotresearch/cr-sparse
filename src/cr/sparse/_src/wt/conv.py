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

from jax import jit
import jax.numpy as jnp

import cr.sparse as crs

def iconv(f, x):
    """Filtering by periodic convolution of x with f
    """
    n = x.shape[0]
    p = f.shape[0]
    x_padded = crs.vec_repeat_at_start(x, p)
    x_filtered = jnp.convolve(x_padded, f)
    return x_filtered[p:n+p]

def aconv(f, x):
    """Filtering by periodic convolution of x with the time reverse of f
    """
    n = x.shape[0]
    p = f.shape[0]
    x_padded = crs.vec_repeat_at_end(x, p)
    # reverse the filter
    f = f[::-1]
    x_filtered = jnp.convolve(x_padded, f)
    return x_filtered[p-1:n+p-1]


def mirror_filter(h):
    """Constructs the mirror filter for a given qmf filter by applying (-1)^t modulation

    Note that modulation starts from -1 rather than 1. This aligns the results with pywt
    """
    n = h.shape[0]
    modulation = (-1)**jnp.arange(1, n+1)
    return modulation * h