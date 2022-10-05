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


"""A port of problem 001 from Sparco
"""

from jax import random
import jax.numpy as jnp


import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.lop as crlop
import cr.nimble.dsp.signals as signals

from .spec import Problem

def generate(key, n=1024):
    name = 'heavi-sine:fourier:heavi-side'
    # time domain signal
    t = jnp.arange(1, n+1) / n
    sig_sine = 4*jnp.sin(4*jnp.pi*t)
    sig_jump = jnp.sign(t - .3) - jnp.sign(.72 - t)
    b = sig_sine + sig_jump
    fourier_basis = crlop.fourier_basis(n)

    dirac_basis = crlop.identity(n)
    heaviside = crlop.heaviside(n)
    inv_heaviside = crlop.inv_heaviside(n)

    # compute the representation of the sine part
    x0 = fourier_basis.trans(sig_sine)
    # compute the representation of the jumps part
    x1 = inv_heaviside.times(sig_jump)
    # combine the representations
    x = jnp.concatenate((x0, x1))
    # combined dictionary for the signal 
    dictionary = crlop.hcat(fourier_basis, heaviside)
    # No sensing matrix
    Phi = dirac_basis
    Psi = dictionary
    A = dictionary
    reconstruct = lambda x : jnp.real(Psi.times(x))

    # Number of figures
    figures = ['HeaviSine signal', 'Sine part of signal', 
    'Piecewise constant component of the HeaviSine signal', 
    'Real part of Fourier coefficients for the Sine component',
    'Imaginary part of Fourier coefficients for the Sine component',
    'Coefficients for the jumps part in Heaviside (non-orthogonal) basis'
    ]
    def plot(i, ax):
        ax.set_title(figures[i])
        if i == 0:
            ax.plot(t, b)
            return
        if i == 1:
            ax.plot(t, sig_sine)
            return
        if i == 2:
            ax.plot(t, sig_jump)
            return
        if i == 3:
            ax.stem(jnp.real(x0))
            return
        if i == 4:
            ax.plot(jnp.imag(x0))
            return
        if i == 5:
            ax.stem(x1)
            return

    return Problem(name=name, Phi=Phi, Psi=Psi, A=A, b=b,
        reconstruct=reconstruct, x=x,
        figures=figures, plot=plot)
