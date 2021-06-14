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

import jax
import jax.numpy as jnp
from jax import random

from .norm import sqr_norms_l2_cw, sqr_norms_l2_rw
from .matrix import is_matrix

def find_first_signal_with_energy_le_rw(X, energy):
    """Returns the index of the first row which has energy less than the specified threshold
    """
    assert is_matrix(X)
    energies = sqr_norms_l2_rw(X)
    index = jnp.argmax(energies <= energy)
    return index if energies[index] <= energy else jnp.array(-1)

def find_first_signal_with_energy_le_cw(X, energy):
    """Returns the index of the first column which has energy less than the specified threshold
    """
    assert is_matrix(X)
    energies = sqr_norms_l2_cw(X)
    index = jnp.argmax(energies <= energy)
    return index if energies[index] <= energy else jnp.array(-1)


def randomize_rows(key, X):
    """Randomizes the rows in X
    """
    assert is_matrix(X)
    m, n = X.shape
    r = random.permutation(key, m)
    return X[r, :]

def randomize_cols(key, X):
    """Randomizes the columns in X
    """
    assert is_matrix(X)
    m, n = X.shape
    r = random.permutation(key, n)
    return X[:, r]


def largest_indices(x, K):
    """Returns the indices of K largest entries in x by magnitude
    """
    indices = jnp.argsort(jnp.abs(x))
    return indices[:-K-1:-1]

def largest_indices_rw(X, K):
    """Returns the indices of K largest entries by magnitude in each row of X
    """
    indices = jnp.argsort(jnp.abs(X), axis=1)
    return indices[:, :-K-1:-1]

def largest_indices_cw(X, K):
    """Returns the indices of K largest entries by magnitude in each column of X
    """
    indices = jnp.argsort(jnp.abs(X), axis=0)
    return indices[:-K-1:-1, :]

def take_along_rows(X, indices):
    """Picks K entries from each row of X specified by indices matrix
    """
    return jnp.take_along_axis(X, indices, axis=1)

def take_along_cols(X, indices):
    """Picks K entries from each column of X specified by indices matrix
    """
    return jnp.take_along_axis(X, indices, axis=0)

def sparse_approximation(x, K):
    """Keeps only largest K non-zero entries by magnitude in a vector x
    """
    if K == 0:
        return x.at[:].set(0)
    indices = jnp.argsort(jnp.abs(x))
    #print(x, K, indices)
    return x.at[indices[:-K]].set(0)
    
def sparse_approximation_cw(X, K):
    #return jax.vmap(sparse_approximation, in_axes=(1, None), out_axes=1)(X, K)
    """Keeps only largest K non-zero entries by magnitude in each column of X
    """
    if K == 0:
        return X.at[:].set(0)
    indices = jnp.argsort(jnp.abs(X), axis=0)
    for c in range(X.shape[1]):
        ind = indices[:-K, c]
        X = X.at[ind, c].set(0)
    return X

def sparse_approximation_rw(X, K):
    """Keeps only largest K non-zero entries by magnitude in each row of X
    """
    if K == 0:
        return X.at[:].set(0)
    indices = jnp.argsort(jnp.abs(X), axis=1)
    for r in range(X.shape[0]):
        ind = indices[r, :-K]
        X = X.at[r, ind].set(0)
    return X


def build_signal_from_indices_and_values(length, indices, values):
    """Builds a sparse signal from its non-zero entries (specified by their indices and values)
    """
    x = jnp.zeros(length)
    indices = jnp.asarray(indices)
    values = jnp.asarray(values)
    return x.at[indices].set(values)


def nonzero_values(x):
    """Returns the values of non-zero entries in x
    """
    return x[x != 0]

def nonzero_indices(x):
    """Returns the indices of non-zero entries in x
    """
    return jnp.nonzero(x)[0]


def hard_threshold(x, K):
    """Returns the indices and corresponding values of largest K non-zero entries in a vector x
    """
    indices = jnp.argsort(jnp.abs(x))
    I = indices[:-K-1:-1]
    x_I = x[I]
    return I, x_I

def hard_threshold_sorted(x, K):
    """Returns the sorted indices and corresponding values of largest K non-zero entries in a vector x
    """
    # Sort entries in x by their magnitude
    indices = jnp.argsort(jnp.abs(x))
    # Pick the indices of K-largest (magnitude) entries in x (from behind)
    I = indices[:-K-1:-1]
    # Make sure that indices are sorted in ascending order
    I = jnp.sort(I)
    # Pick corresponding values
    x_I = x[I]
    return I, x_I


def dynamic_range(x):
    """Returns the ratio of largest and smallest values (by magnitude) in x (dB)
    """
    x = jnp.sort(jnp.abs(x))
    return 20 * jnp.log10(x[-1] / x[0])


def nonzero_dynamic_range(x):
    """Returns the ratio of largest and smallest non-zero values (by magnitude) in x (dB)
    """
    x = nonzero_values(x)
    return dynamic_range(x)
