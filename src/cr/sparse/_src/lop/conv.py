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
Convolutions 1D, 2D, ND
"""
import jax.numpy as jnp


from .impl import _hermitian
from .lop import Operator
from .util import apply_along_axis

def convolve(n, h, offset=0, axis=0):
    """Implements a convolution operator with the filter h

    Note:

        The filter coefficients h are padded with zeros based on the
        offset to ensure that the impulse response of the convolution
        starts from index 0 after SAME convolution.
    """
    m = len(h)
    start = m // 2
    offset = 2 * (start  - int(offset))
    if m % 2 == 0:
        # we need less offset for even length filters
        offset -= 1
    left_pad = offset if offset > 0 else 0
    right_pad = -offset if offset < 0 else 0
    h = jnp.pad(h, (left_pad, right_pad), mode='constant')
    h_conj = _hermitian(h[::-1])
    times1d = lambda x : jnp.convolve(x, h, 'same')
    trans1d = lambda x : jnp.convolve(x, h_conj, 'same')
    times, trans = apply_along_axis(times1d, trans1d, axis)
    return Operator(times=times, trans=trans, shape=(n,n))

def convolve2D():
    pass

def convolveND():
    pass
