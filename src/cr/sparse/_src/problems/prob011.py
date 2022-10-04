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


"""A port of problem 007 from Sparco
"""


from jax import random
import jax.numpy as jnp


import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.lop as crlop

from .spec import Problem

def generate(key, k=32, m=256, n=1024, scale=1):
    name = 'gaussian-spikes:dirac:gaussian'

    k = max(1, round(scale*k))
    m = max(1, round(scale*m))
    n = max(1, round(scale*n))
    keys = random.split(key, 3)

    # Gaussian spikes
    values = random.normal(keys[0], shape=(k,))
    indices = random.choice(keys[1], n, shape=(k,), replace=False)
    x = jnp.zeros(n).at[indices].set(values)

    # sensing matrix
    Phi = crlop.gaussian_dict(keys[2], m, n)
    # measurements
    b = Phi.times(x)
    # sparsifying basis
    Psi = crlop.identity(n)
    # combined operator
    A = Phi
    # function to construct the signal from representation
    reconstruct = lambda x : x

    # Number of figures
    figures = ['Gaussian Spikes', 'Measurements']
    def plot(i, ax):
        ax.set_title(figures[i])
        if i == 0:
            ax.plot(x, 'b-')
            return
        if i == 1:
            ax.plot(b, 'b-')
            return

    return Problem(name=name, Phi=Phi, Psi=Psi, A=A, b=b,
        reconstruct=reconstruct, x=x,
        figures=figures, plot=plot)
