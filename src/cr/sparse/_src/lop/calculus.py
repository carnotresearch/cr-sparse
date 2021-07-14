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

from functools import partial

import jax.numpy as jnp

from .impl import _hermitian
from .lop import Operator


FORWARD_DERIVATIVE_FILTER = jnp.array([1., -1.])

def _derivative_fwd(x, dx):
    append = jnp.array([x[-1]])
    return jnp.diff(x, append=append) / dx

def _derivative_fwd_adj(x, dx):
    x = x.at[-1].set(0)
    prepend = jnp.array([0])
    return jnp.diff(-x, prepend=prepend) / dx

def _derivative_bwd(x, dx):
    prepend = jnp.array([x[0]])
    return jnp.diff(x, prepend=prepend) / dx

def _derivative_bwd_adj(x, dx):
    x = x.at[0].set(0)
    append = jnp.array([0])
    return jnp.diff(-x, append=append) / dx

def _derivative_centered(x, dx):
    diffs = (0.5 * x[2:] - 0.5 * x[:-2]) / dx
    return jnp.pad(diffs, (1,1))

def _derivative_centered_adj(x, dx):
    y = jnp.zeros(x.shape)
    y = y.at[0:-2].add(-0.5*x[1:-1])
    y = y.at[2:].add(0.5*x[1:-1])
    return y


def first_derivative(n, dx=1., kind='centered'):
    """Computes the first derivative
    """
    if kind == 'forward':
        times = partial(_derivative_fwd, dx=dx)
        trans = partial(_derivative_fwd_adj, dx=dx)
    elif kind == 'backward':
        times = partial(_derivative_bwd, dx=dx)
        trans = partial(_derivative_bwd_adj, dx=dx)
    elif kind == 'centered':
        times = partial(_derivative_centered, dx=dx)
        trans = partial(_derivative_centered_adj, dx=dx)
    else:
        raise NotImplemented
    return Operator(times=times, trans=trans, shape=(n,n))


def second_derivative(n, dx=1.):
    filter = jnp.array([1., -2., 1.]) / dx / dx
    times = lambda x : jnp.pad(jnp.convolve(x, filter, 'valid'), (1,1))
    trans = lambda x : jnp.convolve(x[1:-1], filter, 'full')
    return Operator(times=times, trans=trans, shape=(n,n))

