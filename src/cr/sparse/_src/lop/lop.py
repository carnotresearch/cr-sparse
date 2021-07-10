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

from typing import NamedTuple, List, Dict
import jax
import jax.numpy as jnp


class LinearOperator(NamedTuple):
    times : jnp.ndarray
    trans : jnp.ndarray
    m : int
    n : int


def jit(operator):
    times = jax.jit(operator.times)
    trans = jax.jit(operator.trans)
    return LinearOperator(times=times, trans=trans, m=operator.m, n=operator.n)

def identity(m, n):
    times = lambda x:  x
    trans = lambda x : x
    return LinearOperator(times=times, trans=trans, m=m, n=n)


def matrix(A):
    m, n = A.shape
    times = lambda x: A @ x
    trans = lambda x : (x.T @ A ).T
    return LinearOperator(times=times, trans=trans, m=m, n=n)


def hcat(A, B):
    ma, na = A.m, A.n
    mb, nb = B.m, B.n
    assert ma == mb
    m = ma
    n = na + nb
    times = lambda x: jnp.hstack((A.times(x[:na]), B.times(x[na:])))
    trans = lambda x: jnp.hstack((A.trans(x), B.trans(x)))
    return LinearOperator(times=times, trans=trans, m=m, n=n)
