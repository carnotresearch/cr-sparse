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

import jax.numpy as jnp

from .impl import _hermitian
from .lop import Operator
from .util import apply_along_axis

def running_average(n, length, axis=0):
    """Computes a running average of entries in x
    """
    start = length // 2
    filter = jnp.ones(length) / length
    times1d = lambda x : jnp.convolve(x, filter, 'same')
    trans1d = lambda x : jnp.convolve(x, filter, 'full')[start:start+n]
    times, trans = apply_along_axis(times1d, trans1d, axis)
    return Operator(times=times, trans=trans, shape=(n,n))


def fir_filter(n, h, axis=0):
    """Implements an FIR filter defined by coeffs
    """
    h_conj = _hermitian(h[::-1])
    m = len(h)
    start = m // 2
    times1d = lambda x : jnp.convolve(x, h, 'same')
    trans1d = lambda x : jnp.convolve(x, h_conj, 'full')[start:start+n]
    times, trans = apply_along_axis(times1d, trans1d, axis)
    return Operator(times=times, trans=trans, shape=(n,n))

