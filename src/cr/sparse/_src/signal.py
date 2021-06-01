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

import jax
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


def largest_indices(x, K):
    indices = jnp.argsort(jnp.abs(x))
    return indices[:-K-1:-1]

def largest_indices_rw(X, K):
    indices = jnp.argsort(jnp.abs(X), axis=1)
    return indices[:, :-K-1:-1]

def largest_indices_cw(X, K):
    indices = jnp.argsort(jnp.abs(X), axis=0)
    return indices[:-K-1:-1, :]

def take_along_rows(X, indices):
    return jnp.take_along_axis(X, indices, axis=1)

def take_along_cols(X, indices):
    return jnp.take_along_axis(X, indices, axis=0)

def sparse_approximation(x, K):
    if K == 0:
        return x.at[:].set(0)
    indices = jnp.argsort(jnp.abs(x))
    #print(x, K, indices)
    return x.at[indices[:-K]].set(0)
    
def sparse_approximation_cw(X, K):
    #return jax.vmap(sparse_approximation, in_axes=(1, None), out_axes=1)(X, K)
    if K == 0:
        return X.at[:].set(0)
    indices = jnp.argsort(jnp.abs(X), axis=0)
    for c in range(X.shape[1]):
        ind = indices[:-K, c]
        X = X.at[ind, c].set(0)
    return X

def sparse_approximation_rw(X, K):
    if K == 0:
        return X.at[:].set(0)
    indices = jnp.argsort(jnp.abs(X), axis=1)
    for r in range(X.shape[0]):
        ind = indices[r, :-K]
        X = X.at[r, ind].set(0)
    return X
