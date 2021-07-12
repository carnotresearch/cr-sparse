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


def fourier_basis_1d(n):
    """Returns an operator which represents the DFT orthonormal basis
    
    Forward operation is akin to computing inverse fast fourier transform
    scaled by :math:`\\sqrt{n}`

    Adjoint operation is akin to computing forward fast fourier transform
    scaled by :math:`1/\\sqrt{n}`
    """
    n2 = jnp.sqrt(n)
    n3 = 1/n2
    times = lambda x:  n2*jnp.fft.ifft(x, n, axis=0)
    trans = lambda x : n3*jnp.fft.fft(x, n, axis=0)
    return Operator(times=times, trans=trans, shape=(n,n))


def dirac_fourier_basis_1d(n):
    """Returns an operator for a two-ortho basis dictionary consisting of Dirac basis and Fourier basis
    """
    n2 = jnp.sqrt(n)
    n3 = 1/n2
    times = lambda x:  x[:n] + n2*jnp.fft.ifft(x[n:], n, axis=0)
    trans = lambda x : jnp.concatenate((x, n3*jnp.fft.fft(x, n, axis=0)), axis=0)
    return Operator(times=times, trans=trans, shape=(n,2*n))
