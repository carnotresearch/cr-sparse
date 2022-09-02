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

def indicator_l2_ball(q=1., b=None, A=None):
    r"""Returns an indicator function for the closed ball :math:`\| A x - b \|_2 \leq q`

    Args:
        q (float) : Radius of the ball
        b (jax.numpy.ndarray): A vector :math:`b \in \RR^{m}`
        A (jax.numpy.ndarray): A matrix :math:`A \in \RR^{m \times n}`

    Returns:
        An indicator function

    The indicator function is defined as:

    .. math::

        I(x) = \begin{cases} 
            0 & \text{if } \| A x - b \|_2 \leq q \\
            \infty & \text{otherwise}
        \end{cases}

    Special cases:

    * ``indicator_l2_ball()`` returns the Euclidean unit ball :math:`\| x \|_2 \leq 1`.
    * ``indicator_l2_ball(q)`` returns the Euclidean ball :math:`\| x \|_2 \leq q`.
    * ``indicator_l2_ball(q, b=b)`` returns the Euclidean ball at center :math:`b`, :math:`\| x  - b\|_2 \leq q`.

    Notes:

    * If center :math:`b \in \RR^m` is unspecified, we assume the center to be at origin.
    * If radius :math:`q` is unspecified, we assume the radius to be 1.
    * If the matrix :math:`A` is unspecified, we assume :math:`A` to be the identity matrix
      :math:`I \in \RR^{n \times n}`.
    """
    if b is not None:
        b = jnp.asarray(b)
        b = cnb.promote_arg_dtypes(b)

    if A is not None:
        A = jnp.asarray(A)
        A = cnb.promote_arg_dtypes(A)

    if q <= 0:
        raise ValueError("q must be greater than 0")

    @jit
    def indicator_q(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        invalid = norm(x) > q
        return jnp.where(invalid, jnp.inf, 0)

    if b is None and A is None:
        return indicator_q


    @jit
    def indicator_q_b(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # compute difference from center
        r = x - b
        invalid = norm(r) > q
        return jnp.where(invalid, jnp.inf, 0)

    if A is None:
        return indicator_q_b

    if b is None:
        # we have q and A specified. 
        # default value for b
        b = 0.

    @jit
    def indicator_q_b_A(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # compute the residual vector
        r = A @ x - b
        invalid = norm(r) > q
        return jnp.where(invalid, jnp.inf, 0)
    
    return indicator_q_b_A


def indicator_l1_ball(q=1., b=None, A=None):
    r"""Returns an indicator function for the closed l1 ball :math:`\| A x - b \|_1 \leq q`

    Args:
        q (float) : Radius of the ball
        b (jax.numpy.ndarray): A vector :math:`b \in \RR^{m}`
        A (jax.numpy.ndarray): A matrix :math:`A \in \RR^{m \times n}`

    Returns:
        An indicator function

    The indicator function is defined as:

    .. math::

        I(x) = \begin{cases} 
            0 & \text{if } \| A x - b \|_1 \leq q \\
            \infty & \text{otherwise}
        \end{cases}

    Special cases:

    * ``indicator_l1_ball()`` returns the l1 unit ball :math:`\| x \|_1 \leq 1`.
    * ``indicator_l1_ball(q)`` returns the l1 ball :math:`\| x \|_1 \leq q`.
    * ``indicator_l1_ball(q, b=b)`` returns the l1 ball at center :math:`b`, :math:`\| x  - b\|_1 \leq q`.

    Notes:

    * If center :math:`b \in \RR^m` is unspecified, we assume the center to be at origin.
    * If radius :math:`q` is unspecified, we assume the radius to be 1.
    * If the matrix :math:`A` is unspecified, we assume :math:`A` to be the identity matrix
      :math:`I \in \RR^{n \times n}`.
    """

    if b is not None:
        b = jnp.asarray(b)
        b = cnb.promote_arg_dtypes(b)

    if A is not None:
        A = jnp.asarray(A)
        A = cnb.promote_arg_dtypes(A)

    # TODO: This creates problems in JIT
    # assert q > 0, ValueError("q must be greater than 0")

    @jit
    def indicator_q(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        invalid = cnb.arr_l1norm(x) > q
        return jnp.where(invalid, jnp.inf, 0)

    if b is None and A is None:
        return indicator_q


    @jit
    def indicator_q_b(x):
        # compute difference from center
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        r = x - b
        invalid = cnb.arr_l1norm(r) > q
        return jnp.where(invalid, jnp.inf, 0)

    if A is None:
        return indicator_q_b

    if b is None:
        # we have q and A specified. 
        # default value for b
        b = 0.

    @jit
    def indicator_q_b_A(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # compute the residual vector
        r = A @ x - b
        invalid = cnb.arr_l1norm(r) > q
        return jnp.where(invalid, jnp.inf, 0)
    
    return indicator_q_b_A
