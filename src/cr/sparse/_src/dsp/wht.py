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

"""
Fast Walsh Hadamard Transforms
"""

from jax import lax, jit
import jax.numpy as jnp


def fwht_ref(X):
    """Computes the Fast Walsh Hadamard Transform over columns of :math:`X`
    """
    n = X.shape[0]
    # number of stages
    s = (n-1).bit_length()

    def init():
        Y = jnp.empty(X.shape, dtype=X.dtype)
        A  = X[0::2]
        B = X[1::2]
        Y = Y.at[0::2].set(A + B)
        Y = Y.at[1::2].set(A - B)
        return (Y, 1, 2, 4)

    def body(state):
        # gap between x entries
        # number of x entries
        X, count, gap, step = state
        Y = jnp.empty(X.shape, dtype=X.dtype)
        J = 0
        k = 0
        while k < n -1:
            for j in range(J, J+gap-1, 2):
                # compute the four parts
                a = X[j]
                b = X[j+gap]
                c = X[j+1]
                d = X[j+1+gap]
                Y = Y.at[k].set(a+b)
                Y = Y.at[k+1].set(a-b)
                Y = Y.at[k+2].set(c-d)
                Y = Y.at[k+3].set(c+d)
                k += 4
            J += step
        return (Y, count+1, 2*gap, 2*step)

    def cond(state):
        count = state[1]
        return count < s

    state = init()
    while cond(state):
        state = body(state)
    #state = lax.while_loop(cond, body, init())
    return state[0]

@jit
def fwht(X):
    """Computes the Fast Walsh Hadamard Transform over columns of :math:`X`
    """
    n = X.shape[0]
    # number of stages
    s = (n-1).bit_length()

    def init1():
        Y = jnp.empty(X.shape, dtype=X.dtype)
        A  = X[0::2]
        B = X[1::2]
        Y = Y.at[0::2].set(A + B)
        Y = Y.at[1::2].set(A - B)
        return (Y, 1, 2, 4)

    def body1(state):
        # gap between x entries
        # number of x entries
        X, count, gap, step = state
        Y = jnp.empty(X.shape, dtype=X.dtype)
        J = 0
        k = 0
        def body2(state):
            Y, J, k = state
            def body3(state):
                Y, j, k = state
                # compute the four parts
                a = X[j]
                b = X[j+gap]
                c = X[j+1]
                d = X[j+1+gap]
                Y = Y.at[k].set(a+b)
                Y = Y.at[k+1].set(a-b)
                Y = Y.at[k+2].set(c-d)
                Y = Y.at[k+3].set(c+d)
                return (Y, j+2, k+4)
            def cond3(state):
                j = state[1]
                return j <  J+gap-1
            # the loop
            init3 = (Y, J, k)
            Y, j, k = lax.while_loop(cond3, body3, init3)
            return (Y, J + step, k)

        def cond2(state):
            k = state[2]
            return k < n - 1

        init2 = Y, J, 0
        Y, J, k = lax.while_loop(cond2, body2, init2)

        return (Y, count+1, 2*gap, 2*step)

    def cond1(state):
        count = state[1]
        return count < s

    state = lax.while_loop(cond1, body1, init1())
    return state[0]
