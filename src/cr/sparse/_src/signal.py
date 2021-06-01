# Copyright 2021 Carnot Research Pvt Ltd
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
from jax import random

from .norm import sqr_norms_l2_cw, sqr_norms_l2_rw

def find_first_signal_with_energy_le_rw(X, energy):
    energies = sqr_norms_l2_rw(X)
    index = jnp.argmax(energies <= energy)
    return index if energies[index] <= energy else jnp.array(-1)

def find_first_signal_with_energy_le_cw(X, energy):
    energies = sqr_norms_l2_cw(X)
    index = jnp.argmax(energies <= energy)
    return index if energies[index] <= energy else jnp.array(-1)


def randomize_rows(key, X):
    m, n = X.shape
    r = random.permutation(key, m)
    return X[r, :]

def randomize_cols(key, X):
    m, n = X.shape
    r = random.permutation(key, n)
    return X[:, r]
