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
import jax.numpy.fft as jfft


from .impl import _hermitian
from .lop import Operator

import cr.sparse as crs

# from .util import apply_along_axis


def fft(m, n=None, axis=0, mode='r2c'):
    """1D FFT operator
    """
    if n is None: 
        n = m
    assert n >= m, "smaller n is not supported."

    if mode == 'r2c':
        def times(x):
            # make sure that x is real
            x = jnp.real(x)
            # we need complex and conjugate symmetric output
            w = jfft.fft(x, n, axis=axis)
            return w

        def trans(w):
            x = n * jnp.fft.ifft(w, n, axis=axis)
            # we need to return only first m values
            return x[:m]
    elif mode == 'c2c':
        def times(x):
            w = jfft.fft(x, n, axis=axis)
            return w

        def trans(w):
            x = n * jnp.fft.ifft(w, n, axis=axis)
            # we need to return only first m values
            return x[:m]
    else: 
        raise ValueError(f"Unsupported mode {mode}")

    return Operator(times=times, trans=trans, shape=(n,m), real=False)

