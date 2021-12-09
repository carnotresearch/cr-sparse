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

from jax import jit, lax

import jax.numpy as jnp
import cr.nimble as cnb
import cr.sparse.opt as opt

from .prox import build, build_from_ind_proj

@jax.jit
def replace_increasing_subsequences_with_averages(x):
  mask = jnp.zeros(len(x), dtype=bool).at[1:].set(jnp.diff(x) > 0)
  segment_ids = jnp.cumsum(~mask) - 1
  sums = jax.ops.segment_sum(x, segment_ids, num_segments=len(x))
  norms = jax.ops.segment_sum(jnp.ones_like(x), segment_ids, num_segments=len(x))
  return (sums / norms)[segment_ids]



def prox_sorted_l1(lambda_):
    r"""Returns a prox-capable wrapper for the sorted and weighted l1-norm function ``f(x) = sum(lambda * sort(abs(x), 'descend'))``

    Args:
        lambda_ (jax.numpy.ndarray): A strictly positive vector which is sorted in decreasing order

    Returns:
       ProxCapable: A prox-capable function 

    The function is computed in following steps:

    - Take absolute values of entries in x
    - Sort the entries of x in descending order
    - Multiply the sorted entries with entries in lambda (component wise)
    - Compute the sum of the entries

    """
    lambda_ = jnp.asarray(lambda_)
    lambda_ = jnp.ravel(lambda_)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # take absolute values
        x = jnp.abs(x)
        # sort the entries in ascending order
        x = jnp.sort(x)
        # reverse the order
        x = x[::-1]
        # compute element wise product
        x = lambda_ * x
        # return the sum
        return jnp.sum(x)

    def proximal_op(x, t):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)

    return build(func, proximal_op)
