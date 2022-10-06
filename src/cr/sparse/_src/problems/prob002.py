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


"""A port of problem 002 from Sparco
"""

from jax import random
import jax.numpy as jnp


import cr.nimble as crn
import cr.nimble.dsp.signals as signals
import cr.sparse as crs
import cr.sparse.lop as crlop

from .spec import Problem


def generate(key, n=1024):
    name = 'blocks:haar'
    # time domain signal
    t, b = signals.blocks(n)
    # haar wavelet basis operator
    haar = crlop.dwt(n, wavelet='haar', level=5, basis=True)
    # wavelet basis coefficients
    x = haar.trans(b)
    # identity basis
    dirac_basis = crlop.identity(n)
    Phi = dirac_basis
    Psi = haar
    A = haar
    reconstruct = lambda x : Psi.times(x)

    figures = ['Block signal', 'Haar coefficients of the blocks signal']
    def plot(i, ax):
        ax.set_title(figures[i])
        if i == 0:
            ax.plot(t, b)
            return
        if i == 1:
            ax.stem(t, x, markerfmt='.')
            return

    return Problem(name=name, Phi=Phi, Psi=Psi, A=A, b=b,
        reconstruct=reconstruct, x=x,
        figures=figures, plot=plot)
