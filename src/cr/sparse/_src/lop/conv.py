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

        We don't use padding of coefficients of h. It turns out,
        it is faster to perform a full convolution and then 

    """
    assert n > 0
    m = len(h)
    # The location of center of the filter response should be within it.
    assert offset >= 0
    assert offset < m
    forward = offset
    adjoint = m  - 1 - offset
    h_conj = _hermitian(h[::-1])
    times1d = lambda x : jnp.convolve(x, h, 'full')[forward:forward+n]
    trans1d = lambda x : jnp.convolve(x, h_conj, 'full')[adjoint:adjoint+n]
    times, trans = apply_along_axis(times1d, trans1d, axis)
    return Operator(times=times, trans=trans, shape=(n,n))

def convolve2D():
    pass

def convolveND():
    pass
