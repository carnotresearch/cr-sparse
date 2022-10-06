# Copyright 2022 CR-Suite Development Team
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


"""A port of problem 004 from Sparco
"""

from jax import random
import jax.numpy as jnp


import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.lop as crlop

from .spec import Problem


def generate(key, c=2, k=120, n=1024):
    name = 'complex:sinusoid-spikes:dirac-fourier'
    t = jnp.arange(n)/n
    keys = random.split(key, 4)

    fourier_basis = crlop.fourier_basis(n)
    dirac_basis = crlop.identity(n)
    dirac_fourier_basis = crlop.hcat(dirac_basis, fourier_basis)
    dirac_fourier_basis = crlop.jit(dirac_fourier_basis)

    # cosine part of the signal
    indices = random.choice(keys[0], n, shape=(c,), replace=False)
    coeffs = jnp.zeros(n, dtype=jnp.complex128)
    rkey, ikey = random.split(keys[1])
    rc = random.normal(rkey, shape=(c,))
    ic = random.normal(ikey, shape=(c,))
    cc = rc + 1j * ic
    coeffs = coeffs.at[indices].set(cc)
    coeffs = coeffs * jnp.sqrt(n/2)
    sinusoid = fourier_basis.times(coeffs)

    # spike part of the signal
    indices = random.choice(keys[2], n, shape=(k,), replace=False)
    spikes = jnp.zeros(n, dtype=jnp.complex128)
    rkey, ikey = random.split(keys[3])
    rc = random.normal(rkey, shape=(k,))
    ic = random.normal(ikey, shape=(k,))
    cc = rc + 1j * ic
    spikes = spikes.at[indices].set(cc)

    # combined signal
    b = sinusoid + spikes
    x = jnp.concatenate([spikes, coeffs])
    Phi = dirac_basis
    Psi = dirac_fourier_basis
    A = dirac_fourier_basis
    reconstruct = lambda x : Psi.times(x)

    # Number of figures
    figures = ['Real part of the signal', 'Imaginary part of the signal', 
    'Real part of the coefficients', 'Imaginary part of the coefficients']
    def plot(i, ax):
        ax.set_title(figures[i])
        if i == 0:
            ax.plot(t, jnp.real(b))
            return
        if i == 1:
            ax.plot(t, jnp.imag(b))
            return
        if i == 2:
            ax.stem(jnp.real(x), markerfmt='.')
            return
        if i == 3:
            ax.stem(jnp.imag(x), markerfmt='.')
            return

    return Problem(name=name, Phi=Phi, Psi=Psi, A=A, b=b,
        reconstruct=reconstruct, x=x, y=b,
        figures=figures, plot=plot)



