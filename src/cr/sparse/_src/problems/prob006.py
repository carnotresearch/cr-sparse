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


"""A port of problem 006 from Sparco
"""

from jax import random
import jax.numpy as jnp


import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.lop as crlop

from .spec import Problem


def piecewise_polynomial(key, n, p):
    keys = random.split(key, 2)
    pp = random.uniform(keys[0], shape=(p-1,))
    pp = jnp.append(jnp.insert(pp, 0, 0), 1)
    pp = jnp.sort(pp)
    idx = jnp.round(pp * n).astype(int)
    signal = jnp.zeros(n)
    subkeys = random.split(keys[1], p)
    for i in range(p):
        a = idx[i]
        b = idx[i+1]
        x = jnp.linspace(-3,3, b -a)
        c = jnp.round(random.normal(subkeys[i], shape=(4,)) * 100) / 100
        y = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3
        signal = signal.at[a:b].set(y)
    return signal


def generate(key, p=5, m=600, n=2048, level=5):
    name = 'piecewise-cubic-poly:daubechies:gaussian'
    t = jnp.arange(n)/n
    keys = random.split(key, 2)
    # piece wise polynomial signal
    y = piecewise_polynomial(keys[0], n, p)

    # Daubechies basis
    db_basis = crlop.dwt(n, wavelet='db8', level=level, basis=True)
    db_basis = crlop.jit(db_basis)


    # wavelet representation
    x = db_basis.trans(y)
    # sensing matrix
    Phi = crlop.gaussian_dict(keys[1], m, n)
    # measurements
    b = Phi.times(y)
    # sparsifying basis
    Psi = db_basis
    # combined operator
    A = crlop.compose(Phi, Psi)
    # function to construct the signal from representation
    reconstruct = lambda x : Psi.times(x)

    # Number of figures
    figures = ['Piecewise cubic polynomial', 'Wavelet coefficients', 
    'Measurements']
    def plot(i, ax):
        ax.set_title(figures[i])
        if i == 0:
            ax.plot(y)
            return
        if i == 1:
            ax.stem(x, markerfmt='.')
            return
        if i == 2:
            ax.plot(b)
            return

    return Problem(name=name, Phi=Phi, Psi=Psi, A=A, b=b,
        reconstruct=reconstruct, x=x, y=y,
        figures=figures, plot=plot, both=True)



