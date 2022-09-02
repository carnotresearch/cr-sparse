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

from jax import jit, lax
from jax.ops import segment_sum

import jax.numpy as jnp
import cr.nimble as cnb
import cr.sparse.opt as opt

from .prox import build, build_from_ind_proj

"""
See https://github.com/google/jax/discussions/8862 for the cool trick of averaging of increasing segments
"""


def prox_ordered_l1_b(state):
    y, l = state
    x = y - l
    n = len(x)
    mask = jnp.zeros(len(x), dtype=bool).at[1:].set(jnp.diff(x) > 0)
    segment_ids = jnp.cumsum(~mask) - 1
    y_sums = segment_sum(y, segment_ids, num_segments=n)
    l_sums = segment_sum(l, segment_ids, num_segments=n)
    norms = segment_sum(jnp.ones_like(x), segment_ids, num_segments=n)
    y = (y_sums / norms)[segment_ids]
    l = (l_sums / norms)[segment_ids]
    return y, l


def is_not_nonincreasing(state):
    y, l = state
    return jnp.logical_not(cnb.is_nonincreasing_vec(y -l)) 

def prox_ordered_l1_a(y, l):
    l = jnp.ravel(l)
    # convert them to 1d arrays
    y = jnp.ravel(y)
    # get the sign vector of y
    sgn = jnp.sign(y)
    # take the absolute values
    y = jnp.abs(y)
    # sort entries in y by magnitude
    idx = jnp.argsort(y)
    # go in descending order
    idx = idx[::-1]
    y = y[idx]
    # make sure that lambda and y are in same shape
    l = jnp.broadcast_to(l, y.shape)
    state = y, l
    state = lax.while_loop(is_not_nonincreasing, prox_ordered_l1_b, state)
    y, l = state
    x = (y - l)
    # keep only the postive part
    x = jnp.where(x > 0, x , 0)
    # restore x at the original indices
    x = jnp.zeros_like(x).at[idx].set(x)
    # restore the sign
    x = sgn * x
    return x

def prox_owl1(lambda_ = 1.):
    r"""Returns a prox-capable wrapper for the ordered and weighted l1-norm function ``f(x) = sum(lambda * sort(abs(x), 'descend'))``

    Args:
        lambda_ (jax.numpy.ndarray): A strictly positive vector which is sorted in decreasing order

    Returns:
       ProxCapable: A prox-capable function 

    Let :math:`x \in \RR^n`. Let :math:`|x|` represent a vector of absolute values of entries in :math:`x`.
    Let :math:`|x|_{\downarrow}` represent a vector consisting of entries in :math:`|x|` sorted in descending order.
    Let :math:`|x|_{(1)} \geq |x|_{(2)} \geq |x|_{(3)} \geq \dots \geq |x|_{(n)}` represent the order statistic of :math:`x`,
    i.e. entries in :math:`x` arranged in descending order by magnitude.

    Let :math:`\lambda \in \RR^n_{+}` be a weight vector such that 
    :math:`\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n` and :math:`\lambda \neq 0` i.e. 
    not all entries in :math:`\lambda` are zero.

    Then the ordered weighted :math:`\ell_1` norm of :math:`x` w.r.t. the weight vector :math:`\lambda` is defined as:

    .. math::

        J_{\lambda} (x) = \sum_{1}^n \lambda_i | x |_{(i)}

    The function is computed in following steps:

    - Take absolute values of entries in x
    - Sort the entries of x in descending order
    - Multiply the sorted entries with entries in lambda (component wise)
    - Compute the sum of the entries

    For the derivation of the proximal operator for the ordered and weighted l1 norm, see :cite:`lgorzata2013statistical`.

    """
    lambda_ = jnp.asarray(lambda_)
    lambda_ = cnb.promote_arg_dtypes(lambda_)
    lambda_ = jnp.ravel(lambda_)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # take absolute values
        x = jnp.abs(x)
        # convert x to 1d array
        x = jnp.ravel(x)
        # sort the entries in ascending order
        x = jnp.sort(x)
        # reverse the order
        x = x[::-1]
        # compute element wise product
        x = lambda_ * x
        # return the sum
        return jnp.sum(x)

    @jit
    def proximal_op(x, t):
        # make sure that x is a JAX array
        x = jnp.asarray(x)
        # make sure that x is float
        x = cnb.promote_arg_dtypes(x)
        # capture original shape
        shape = x.shape
        # convert x to 1d array
        x = jnp.ravel(x)
        # compute the proximal vector
        z = prox_ordered_l1_a(x, t*lambda_)
        # put it back into original shape
        z = jnp.reshape(z, shape)
        return z

    return build(func, proximal_op)
