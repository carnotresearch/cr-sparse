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

import jax.numpy as jnp

from .impl import _hermitian
from .lop import Operator



def circulant(n, c):
    r = len(c)
    assert n >= r
    c_padded = jnp.pad(c, (0, n-r))
    cf = jnp.fft.rfft(c_padded)
    c_j = jnp.roll(c_padded[::-1], 1)
    cjf = jnp.fft.rfft(c_j)
    times = lambda x : jnp.fft.irfft(jnp.fft.rfft(x) * cf)
    trans = lambda x : jnp.fft.irfft(jnp.fft.rfft(x) * cjf)
    return Operator(times=times, trans=trans, shape=(n,n))
