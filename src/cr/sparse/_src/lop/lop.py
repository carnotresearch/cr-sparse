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

from typing import NamedTuple, Callable, Tuple
import jax
import jax.numpy as jnp

from .impl import _hermitian

def column(T, i):
    """Returns the i-th column of the operator T
    """
    e = jnp.zeros(T.shape[1]).at[i].set(1.)
    return T.times(e)

column = jax.jit(column, static_argnums=(0, 1))

def columns(T, indices):
    """Returns the i-th column of the operator T
    """
    n = T.shape[1]
    k = len(indices)
    e = jnp.zeros((n, k))
    e = e.at[indices, jnp.arange(k)].set(1.)
    return T.times(e)

columns = jax.jit(columns, static_argnums=(0,))

class Operator(NamedTuple):
    """
    Represents a finite linear operator :math:`T : A -> B` where :math:`A` and :math:`B` are finite vector spaces.

    Parameters:
        times: A function implementing :math:`T(x)`
        trans: A function implementing :math:`T^H (x)`
        m: The dimension of the destination vector space :math:`B`
        n: The dimension of the source vector space :math:`A`
        linear: Indicates if the operator is linear or not
        jit_safe: Indicates if the operator can be safely JIT compiled
        matrix_safe: Indicates if the operator can accept a matrix of vectors
        real: Indicates if a linear operator is real i.e. has a matrix representation of real numbers

    Note:
        While most of the operators in the library are linear operators,
        some are not. Prominent examples include operators
        like real part operator, imaginary part operator.
        These operators are provided for convenience.

        Most operators in this collection are real.
    """
    times : Callable[[jnp.ndarray], jnp.ndarray]
    """A linear function mapping from A to B """
    trans : Callable[[jnp.ndarray], jnp.ndarray]
    """Corresponding adjoint linear function mapping from B to A"""
    shape : Tuple[int, int]
    """Dimension of the linear operator (m, n)"""
    linear : bool = True
    """Indicates if the operator is linear or not"""
    jit_safe: bool = True
    """Indicates if the times and trans functions can be safely jit compiled"""
    matrix_safe: bool = True
    """Indicates if the operator can accept a matrix of vectors"""
    real: bool = True
    """Indicates if a linear operator is real i.e. has a matrix representation of real numbers"""
    column = column
    """Returns a specific column of the matrix representation of the operator"""
    columns = columns
    """Returns a subset of columns of the matrix representation of the operator"""

    def __neg__(self):
        """Returns the nagative of this linear operator"""
        return neg(self)

    def __add__(self, other):
        """Returns the sum of this linear operator with another linear operator"""
        return add(self, other)

    def __sub__(self, other):
        """Returns the subtraction of this linear operator with another linear operator"""
        return subtract(self, other)

    def __matmul__(self, other):
        """Returns the composition of this linear operator with another linear operator"""
        return compose(self, other)

    def __pow__(self, n):
        """Returns a linear operator which works like applying :math:`T` n times"""
        return power(self, n)

    def times_2d(self, X):
        """Computes Y = T X  where y = T x for each column in X"""
        return jax.vmap(self.times, (1), (1))(X)

    def trans_2d(self, Y):
        """Computes Y = T^H X  where y = T^H x for each column in X"""
        return jax.vmap(self.trans, (1), (1))(Y)

    def apply_columns(self, x):
        """Computes y = T_I x where I is an index set selecting  a subset of columns of T"""
        xr = jnp.zeros(x.shape, x.dtype)
        xr = xr.at[I].set(x[I])
        return self.times(xr)


def jit(operator):
    """Returns the same linear operator with compiled times and trans functions"""
    if not operator.jit_safe:
        raise Exception("This operator is not suitable for JIT compilation.")
    times = jax.jit(operator.times)
    trans = jax.jit(operator.trans)
    return Operator(times=times, trans=trans, shape=operator.shape, matrix_safe=operator.matrix_safe)

###########################################################################################
#
#  Operator algebra
#
###########################################################################################

###########################################################################################
#  Unary operations on linear operators
###########################################################################################

def neg(A):
    """Returns the negative of a linear operator :math:`T = -A`"""
    times = lambda x : -A.times(x)
    trans = lambda x : -A.trans(x)
    return Operator(times=times, trans=trans, shape=A.shape, jit_safe=A.jit_safe, matrix_safe=A.matrix_safe, real=A.real)

def scale(A, alpha):
    """Returns the linear operator :math:`T = \\alpha A` for the operator :math:`A`"""
    real = A.real and not  isinstance(alpha, complex)
    times = lambda x : alpha * A.times(x)
    trans = lambda x : alpha * A.trans(x)
    return Operator(times=times, trans=trans, shape=A.shape, jit_safe=A.jit_safe, matrix_safe=A.matrix_safe, real=real)

def hermitian(A):
    """Returns the Hermitian transpose of a given operator :math:`T = A^H`"""
    m, n = A.shape
    return Operator(times=A.trans, trans=A.times, shape=(n,m), jit_safe=A.jit_safe, matrix_safe=A.matrix_safe, real=A.real)

def transpose(A):
    """Returns the transpose of a given operator :math:`T = A^T`"""
    m, n = A.shape
    if A.real:
        times = A.trans
        trans = A.times
    else:
        times = lambda x: _hermitian(A.trans(_hermitian(x)))
        trans = lambda x: _hermitian(A.times(_hermitian(x)))
    return Operator(times=times, trans=trans, shape=(n,m), jit_safe=A.jit_safe, matrix_safe=A.matrix_safe, real=A.real)

def apply_n(func, n, x):
    init = (x, 0)
    def body(state):
        x, c = state
        return func(x), c+1
    def cond(state):
        return state[1] < n
    state = jax.lax.while_loop(cond, body, init)
    return state[0]

apply_n =  jax.jit(apply_n, static_argnums=(0, 1))

def power(A, p):
    """Returns the linear operator :math:`T = A^p`"""
    m, n = A.shape
    assert m == n
    times = lambda x :apply_n(A.times, p, x)
    trans = lambda x : apply_n(A.trans, p, x)
    return Operator(times=times, trans=trans, shape=A.shape, jit_safe=A.jit_safe, matrix_safe=A.matrix_safe, real=A.real)

def partial_op(A, picks, perm=None):
    """Returns the linear operator T that computes  (A x[perm])[picks]

    We are allowing for two kinds of randomizations
    - entries in the model vector x can be permuted (as per a random perm)
    - From the result y = A x[perm], a limited number of entries can be picked (as per picks)
    """
    assert picks.ndim == 1
    m, n = A.shape
    k = picks.shape[0]
    if perm is not None:
        assert perm.ndim == 1


    if perm is None:
        times = lambda x: (A.times(x))[picks]

        def trans(x):
            # expand the input with zero entries
            shape = (m,) + x.shape[1:]
            tmp = jnp.zeros(shape, dtype=x.dtype)
            tmp = tmp.at[picks].set(x)
            # apply the adjoint
            y = A.trans(tmp)
            return y
    else:
        times = lambda x: (A.times(x[perm]))[picks]

        def trans(x):
            # expand the input with zero entries
            shape = (m,) + x.shape[1:]
            tmp = jnp.zeros(shape, dtype=x.dtype)
            tmp = tmp.at[picks].set(x)
            # apply the adjoint
            y = A.trans(tmp)
            # place the output in correct permutation
            out = jnp.empty(y.shape, dtype=y.dtype)
            out = out.at[perm].set(y)
            return out

    return Operator(times=times, trans=trans, shape=(k, n), jit_safe=A.jit_safe, matrix_safe=A.matrix_safe, real=A.real)


###########################################################################################
#  Binary operations on linear operators
###########################################################################################

def add(A, B):
    """Returns the sum of two linear operators :math:`T = A + B`"""
    ma, na = A.shape
    mb, nb = B.shape
    assert ma == mb
    assert na == nb
    jit_safe = A.jit_safe and B.jit_safe
    matrix_safe = A.matrix_safe and B.matrix_safe
    real = A.real and B.real
    times = lambda x: A.times(x) + B.times(x)
    trans = lambda x: A.trans(x) + B.trans(x)
    return Operator(times=times, trans=trans, shape=A.shape, jit_safe=jit_safe, matrix_safe=matrix_safe, real=real)


def subtract(A, B):
    """Returns a linear operator :math:`T  = A - B`"""
    ma, na = A.shape
    mb, nb = B.shape
    assert ma == mb
    assert na == nb
    jit_safe = A.jit_safe and B.jit_safe
    matrix_safe = A.matrix_safe and B.matrix_safe
    real = A.real and B.real
    times = lambda x: A.times(x) - B.times(x)
    trans = lambda x: A.trans(x) - B.trans(x)
    return Operator(times=times, trans=trans, shape=A.shape, jit_safe=jit_safe, matrix_safe=matrix_safe, real=real)


def compose(A, B):
    """Returns the composite linear operator :math:`T = AB` such that :math:`T(x)= A(B(x))`"""
    ma, na = A.shape
    mb, nb = B.shape
    assert na == mb
    jit_safe = A.jit_safe and B.jit_safe
    matrix_safe = A.matrix_safe and B.matrix_safe
    real = A.real and B.real
    times = lambda x: A.times(B.times(x))
    trans = lambda x: B.trans(A.trans(x))
    return Operator(times=times, trans=trans, shape=(ma, nb), real=real)


def hcat(A, B):
    """Returns the linear operator :math:`T = [A \\, B]`"""
    ma, na = A.shape
    mb, nb = B.shape
    assert ma == mb
    m = ma
    n = na + nb
    jit_safe = A.jit_safe and B.jit_safe
    matrix_safe = A.matrix_safe and B.matrix_safe
    real = A.real and B.real
    times = lambda x: A.times(x[:na]) + B.times(x[na:])
    trans = lambda x: jnp.concatenate((A.trans(x), B.trans(x)))
    return Operator(times=times, trans=trans, shape=(m,n), jit_safe=jit_safe, matrix_safe=matrix_safe, real=real)



