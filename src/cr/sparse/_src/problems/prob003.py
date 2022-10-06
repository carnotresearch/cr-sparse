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


"""A port of problem 003 from Sparco
"""

from jax import random
import jax.numpy as jnp


import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.lop as crlop

from .spec import Problem


def generate(key, c=2, k=120, n=1024):
    name = 'cosine-spikes:dirac-dct'
    t = jnp.arange(n)/n
    keys = random.split(key, 4)

    cosine_basis = crlop.cosine_basis(n)
    dirac_basis = crlop.identity(n)
    dirac_cosine_basis = crlop.hcat(dirac_basis, cosine_basis)
    dirac_cosine_basis = crlop.jit(dirac_cosine_basis)

    # cosine part of the signal
    indices = random.choice(keys[0], n, shape=(c,), replace=False)
    coeffs = jnp.zeros(n)
    coeffs = coeffs.at[indices].set(random.normal(keys[1], shape=(c,)))
    coeffs = coeffs * jnp.sqrt(n/2)
    cosine = cosine_basis.times(coeffs)

    # spike part of the signal
    indices = random.choice(keys[2], n, shape=(k,), replace=False)
    spikes = jnp.zeros(n)
    spikes = spikes.at[indices].set(random.normal(keys[3], shape=(k,)))

    # combined signal
    b = cosine + spikes
    x = jnp.concatenate([spikes, coeffs])
    Phi = dirac_basis
    Psi = dirac_cosine_basis
    A = dirac_cosine_basis
    reconstruct = lambda x : A.times(x)

    # Number of figures
    figures = ['Cosine with spikes', 'Cosine part of signal', 
    'Spikes part of signal', 'Dirac DCT Model']
    def plot(i, ax):
        ax.set_title(figures[i])
        if i == 0:
            ax.plot(t, b)
            return
        if i == 1:
            ax.plot(t, cosine)
            return
        if i == 2:
            ax.stem(t, spikes, markerfmt='.')
            return
        if i == 3:
            ax.stem(x, markerfmt='.')
            return

    return Problem(name=name, Phi=Phi, Psi=Psi, A=A, b=b,
        reconstruct=reconstruct, x=x,
        figures=figures, plot=plot)



