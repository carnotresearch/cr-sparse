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
import jax.numpy.fft as jfft

from .impl import _hermitian
from .lop import Operator

import cr.sparse as crs
import cr.sparse.dsp as crdsp

def fourier_basis(n):
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
    return Operator(times=times, trans=trans, shape=(n,n), real=False)


def dirac_fourier_basis(n):
    """Returns an operator for a two-ortho basis dictionary consisting of Dirac basis and Fourier basis
    """
    n2 = jnp.sqrt(n)
    n3 = 1/n2
    times = lambda x:  x[:n] + n2*jnp.fft.ifft(x[n:], n, axis=0)
    trans = lambda x : jnp.concatenate((x, n3*jnp.fft.fft(x, n, axis=0)), axis=0)
    return Operator(times=times, trans=trans, shape=(n,2*n), real=False)



def cosine_basis(n):
    """Returns an operator which represents the DCT-II orthonormal basis
    
    Forward operation is akin to computing inverse discrete cosine transform
    scaled appropriately

    Adjoint operation is akin to computing forward discrete cosine transform
    scaled appropriately
    """

    factor = jnp.sqrt(2*n)
    ks = jnp.arange(n)

    phi_f = jnp.exp(1j*jnp.pi*ks/(2*n))
    phi_f = phi_f*factor
    phi_f = phi_f.at[0].set(phi_f[0]*jnp.sqrt(2))

    phi_a = jnp.exp(-1j*jnp.pi*ks/(2*n))
    phi_a = phi_a.at[0].set(phi_a[0]*1/jnp.sqrt(2))
    phi_a = phi_a / factor


    def times(x):
        upper = (phi_f*x.T).T
        lower = jnp.zeros((1,)+x.shape[1:])
        c = jnp.concatenate((upper, lower))
        return jfft.irfft(c, axis=0)[:n]

    def trans(x):
        x = jnp.concatenate( (x[:],  x[::-1]))
        c = jfft.rfft(x, axis=0)[:n]
        prod = jnp.real(phi_a*c.T).T
        return prod

    return Operator(times=times, trans=trans, shape=(n,n))

def walsh_hadamard_basis(n):
    """Returns an operator which represents the Walsh Hadamard Transform Basis

    Note:
        This is a self-adjoint operator
    """
    assert crs.is_power_of_2(n), "Only powers of 2 are supported as n"
    factor = 1/jnp.sqrt(n)
    times = lambda x: factor * crdsp.fwht(x)
    trans = times
    return Operator(times=times, trans=trans, shape=(n,n))
