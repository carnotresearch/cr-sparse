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


from jax import jit

import jax.numpy as jnp
from jax.numpy.linalg import qr, norm

import cr.nimble as cnb

def indicator_zero():
    r"""Returns an indicator function for all zero arrays

    Returns:
        An indicator function

    The zero indicator function is defined as:

    .. math::

        I(x) = \begin{cases} 
            0 & \text{if } x = 0 \\
            \infty       & \text{if } x \neq 0
        \end{cases}

    .. note::

        By :math:`0` in the R.H.S. we mean the zero 
        vector :math:`0 \in \RR^n`.  The dimension :math:`n`
        is left unspecified and inferred automatically 
        from the input :math:`x`.

        The 0 on the L.H.S. is a scalar :math:`0 \in \RR`
        as an indicator function is real valued.
    """

    @jit
    def indicator(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        is_nonzero = jnp.any(x != 0)
        return jnp.where(is_nonzero, jnp.inf, 0)
    
    return indicator


def indicator_singleton(c):
    r"""Returns an indicator function for a singleton set :math:`C = \{c\}`

    Args:
        c (jax.numpy.ndarray): An array

    Returns:
        An indicator function


    Let :math:`C` be a singleton convex set :math:`\{ c\}` 
    where :math:`c \in \RR^n`.

    We implement its indicator function as: 

    .. math::

        I(x) = \begin{cases} 
        0 & \text{if } x = c \\
        \infty & \text{if } x \neq c
        \end{cases}

    .. note::

        The implementation broadcasts :math:`c` to the 
        shape of :math:`x` before making the comparison.

        Thus if :math:`c == 4` and :math:`x = [4,4,4,4]`,
        then :math:`I(x) = 0`.

    """
    c = jnp.asarray(c)
    c = cnb.promote_arg_dtypes(c)
    @jit
    def indicator(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        is_nonzero = jnp.any(x - c != 0)
        return jnp.where(is_nonzero, jnp.inf, 0)

    return indicator


def indicator_affine(A, b=0):
    r"""Returns an indicator function for the linear system  :math:`A x = b`

    Args:
        A (jax.numpy.ndarray): A matrix :math:`A \in \RR^{m \times n}`
        b (jax.numpy.ndarray): A vector :math:`b \in \RR^{m}`

    Returns:
        An indicator function

    The indicator function is defined as:

    .. math::

        I(x) = \begin{cases} 
            0 & \text{if } A x = b \\
            \infty & \text{otherwise}
        \end{cases}

    The convex set :math:`C` is an affine space which is 
    the solution set of system of 
    linear equations :math:`A x = b`. It is parallel to 
    the null space of :math:`A`.
    """
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    A, b = cnb.promote_arg_dtypes(A, b)
    @jit
    def indicator(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # compute the residual
        r = A @ x - b
        # compute the strength of residual
        strength = norm(r) / norm(b)
        return jnp.where(strength > 1e-10, jnp.inf, 0)

    return indicator


def indicator_box(l=None, u=None):
    r"""Returns an indicator function for the box :math:`l \preceq  x \preceq u`

    Args:
        l (jax.numpy.ndarray): Element wise lower bound :math:`l \in \RR^{n}`
        u (jax.numpy.ndarray): Element wise upper bound :math:`u \in \RR^{n}`

    Returns:
        An indicator function

    The indicator function is defined as:

    .. math::

        I(x) = \begin{cases} 
            0 & \text{if } l \preceq x \preceq u \\
            \infty & \text{otherwise}
        \end{cases}

    * The convex set :math:`C = \{ x : \; l \preceq x \preceq u \}`..
    * If :math:`l` is not specified, :math:`C = \{ x : \; x \preceq u \}`.
    * If :math:`u` is not specified, :math:`C = \{ x : \; l \preceq x \}`.

    At least lower or upper bound must be specified. Both cannot be left
    unspecified.
    """
    if l is None and u is None:
        raise ValueError("At least lower or upper bound must be defined.")
    if l is not None:
        l = jnp.asarray(l)
        l = cnb.promote_arg_dtypes(l)
    if u is not None:
        u = jnp.asarray(u)
        u = cnb.promote_arg_dtypes(u)

    @jit
    def lower_bound(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        is_invalid = jnp.any(x < l)
        return jnp.where(is_invalid, jnp.inf, 0)

    @jit
    def upper_bound(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        is_invalid = jnp.any(x > u)
        return jnp.where(is_invalid, jnp.inf, 0)


    @jit
    def box_bound(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        is_invalid = jnp.logical_or(jnp.any(x < l), jnp.any(x > u))
        return jnp.where(is_invalid, jnp.inf, 0)

    if l is None:
        return upper_bound

    if u is None:
        return lower_bound

    return box_bound


def indicator_box_affine(l, u, a, alpha=0., tol=1e-6):
    r"""Returns indicator function for the constraints l <= x <= u and a' x = alpha
    """
    if a is None:
        raise ValueError("a is required")
    a = jnp.asarray(a)
    a = cnb.promote_arg_dtypes(a)
    n = a.size
    if l is None:
        l = jnp.full_like(a, -jnp.inf)
    else:
        l = jnp.asarray(l)
        l = cnb.promote_arg_dtypes(l)
    if u is None:
        u = jnp.full_like(a, jnp.inf)
    else:
        u = jnp.asarray(u)
        u = cnb.promote_arg_dtypes(u)

    @jit
    def indicator(x):
        is_invalid = jnp.any(x < l)
        is_invalid = jnp.logical_or(is_invalid, jnp.any(x > u))
        mismatch = jnp.abs(cnb.arr_rdot(a, x) - alpha)
        affine_invalid = mismatch > tol
        is_invalid = jnp.logical_or(is_invalid, affine_invalid)
        return jnp.where(is_invalid, jnp.inf, 0)

    return indicator


def indicator_conic():
    r"""Returns an indicator function for Lorentz/ice-cream cone :math:`{(x,t): \| x \|_2 \leq t}`

    Let :math:`y \in \RR^{n+1}`. Split :math:`y` as :math:`y = (x, t)` where
    :math:`x \in \RR^n` and :math:`t` is the last (scalar) entry in :math:`y`.

    We then define the convex set :math:`C \subset \RR^{n+1}` as
    :math:`C = \{ y = (x,t) : \; \|x \|_2 \leq t  \}`.

    The indicator function is defined as:

    .. math::

        I((x,t)) = \begin{cases} 
            0 & \text{if } \| x \|_2 \leq t \\
            \infty & \text{otherwise}
        \end{cases}
    
    The ice-cream cone doesn't include any point with :math:`t \lt 0`.
    """
    @jit
    def indicator(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        x, t = x[:-1], x[-1]
        inside = norm(x) <= t
        return jnp.where(inside, 0, jnp.inf)

    return indicator
