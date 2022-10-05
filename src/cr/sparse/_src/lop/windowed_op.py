# Copyright 2021 CR-Suite Development Team
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

import math

from jax import lax
import jax.numpy as jnp
from .lop import Operator
from .util import apply_along_axis

def windowed_op(m, T, overlap=0, axis=0):
    """A wrapper to convert an operator into an overcomplete windowed operator

    Args:
        m (int): Length of output
        T (Operator): The linear operator to be wrapped
        overlap (int): The amount of overlap between two windows

    This operator scans the input window by window, applies T on
    each window and then concatenates the results.

    * The window length is determined on the basis of the input size
      for T.
    * The length of input (n) is determined on the window and overlap sizes.
    * overlap must be less than window length. 

    If overlap is zero, then this operator behaves like a block diagonal
    operator which each block is processed by T.
    """
    # The shape of the underlying operator
    tm, tn = T.shape
    # window length
    w = tm
    assert overlap < w, "Overlap must be less than window size"
    offset = w - overlap
    # number of blocks
    n_blocks = max(1, math.ceil((m - w) / offset) + 1)
    # input length
    n = n_blocks * tn
    real = T.real
    dtype = jnp.float64 if real else jnp.complex128

    yl = tm + (n_blocks - 1) * offset
    m_range = jnp.arange(tm)
    n_range = jnp.arange(tn)
    # initial value of y for times operation
    yf = jnp.zeros(yl, dtype=dtype)
    # x padding to be used in adjoint operation
    xz = jnp.zeros(tm + yl -m)
    # initial value of y for trans operation
    ya = jnp.zeros(tn * n_blocks)

    def times1d(x):
        def body_func(i, y):            
            xw = x[i*tn + n_range]
            idx = i * offset + m_range
            return y.at[idx].add(T.times(xw))
        y = lax.fori_loop(0, n_blocks, body_func, yf)
        return y[:m]


    def trans1d(x):
        # pad x with zeros
        x = jnp.concatenate([x, xz])
        def body_func(i, y):
            xw = x[i * offset + m_range]
            return y.at[i*tn + n_range].set(T.trans(xw))
        return lax.fori_loop(0, n_blocks, body_func, ya)

    times, trans = apply_along_axis(times1d, trans1d, axis)

    return Operator(times=times, trans=trans, shape=(m,n), real=real)
