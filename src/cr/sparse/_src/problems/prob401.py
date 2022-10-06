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


"""A port of problem 401 from Sparco
"""


from jax import random
import jax.numpy as jnp

from scipy.io import wavfile

import cr.nimble as crn
from cr.nimble.io.resource import ensure_cr_suite_resource
import cr.sparse as crs
import cr.sparse.lop as crlop

from .spec import Problem


MIXING_DEFAULT = jnp.array([[0.1375, 0.7250],
[0.9905, 0.6888]])

def generate(key, mixing=MIXING_DEFAULT):
    name = 'src-sep-1'
    
    # Read the guitar wave file
    guitar_path = ensure_cr_suite_resource('sparco/prob401_Guitar.wav')
    g_rate, guitar = wavfile.read(guitar_path)
    # Downsample from 48.0kHz to 8.0kHz
    guitar = guitar[::6]

    # Read the piano wave file
    piano_path = ensure_cr_suite_resource('sparco/prob401_Piano.wav')
    p_rate, piano = wavfile.read(piano_path)
    # Downsample from 44.1kHz to 8.8kHz
    piano = piano[::5, :]

    # Extract part
    guitar = jnp.ravel(guitar)
    piano = jnp.ravel(piano)
    piano = piano[5000:5000+len(guitar)]
    m = len(guitar)

    # stack them together in columns
    signal = jnp.column_stack([guitar, piano])
    # prepare the mixture
    mixture = signal @ mixing.T

    wlen = 512
    overlap = 256
    # Windowed DCT basis operator
    cosine_basis = crlop.cosine_basis(wlen)
    Psi = crlop.windowed_op(m, cosine_basis, overlap)
    # Mixing operator
    Phi = crlop.matrix(mixing, axis=1)
    # combined operator
    A = crlop.compose(Phi, Psi, ignore_compatibility=True)
    # function to construct the signal from representation
    reconstruct = lambda x : Psi.times(x)
    Phi = crlop.jit(Phi)
    Psi = crlop.jit(Psi)
    A = crlop.jit(A)

    # Number of figures
    figures = ['Audio signal 1 (Guitar)', 'Audio signal 2 (Piano)',
    'Mixed audio signal 1', 'Mixed audio signal 2']
    def plot(i, ax):
        ax.set_title(figures[i])
        if i == 0:
            ax.plot(signal[:, 0])
            return
        if i == 1:
            ax.plot(signal[:, 1])
            return
        if i == 2:
            ax.plot(mixture[:, 0])
            return
        if i == 3:
            ax.plot(mixture[:, 1])
            return

    return Problem(name=name, Phi=Phi, Psi=Psi, A=A, b=mixture,
        reconstruct=reconstruct, y=signal,
        figures=figures, plot=plot)
