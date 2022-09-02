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

import jax.numpy as jnp

from .lop import Operator

from .util import apply_along_axis

def identity(in_dim, out_dim=None, axis=0):
    """Returns an identity linear operator from model space to data space

    Args:
        in_dim (int): Dimension of the model space 
        out_dim (int): Dimension of the data space
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: An identity linear operator

    If ``out_dim`` is not specified, then we assume that 
    both model space and data space have same dimension.

    Example:
        A square identity operator::

            >>> T = lop.identity(4)
            >>> T.times(jnp.arange(4) + 0.)
            DeviceArray([0., 1., 2., 3.], dtype=float32)
            >>> T.trans(jnp.arange(4))
            DeviceArray([0, 1, 2, 3], dtype=int32)

        A tall identity operator (output has more dimensions)::

            >>> T = lop.identity(4, 6)
            >>> T.times(jnp.arange(4) + 0.)
            DeviceArray([0., 1., 2., 3., 0., 0.], dtype=float32)
            >>> T.trans(T.times(jnp.arange(4) + 0.))
            DeviceArray([0., 1., 2., 3.], dtype=float32)

        A wide identity operator (output has less dimensions)::

            >>> T = lop.identity(4, 3)
            >>> T.times(jnp.arange(4) + 0.)
            DeviceArray([0., 1., 2.], dtype=float32)
            >>> T.trans(T.times(jnp.arange(4) + 0.))
            DeviceArray([0., 1., 2., 0.], dtype=float32)

        By default T applies along columns of a matrix (axis=0)::

            >>> T.times(jnp.arange(20).reshape(4, 5))
            DeviceArray([[ 0,  1,  2,  3,  4],
             [ 5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14]], dtype=int32)

        Identity operator applying along rows of a 2D matrix::

            >>> T = lop.identity(4, 3, axis=1)
            >>> T.times(jnp.arange(20).reshape(5, 4))
            DeviceArray([[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]], dtype=int32)
    """
    out_dim = in_dim if out_dim is None else out_dim

    if in_dim == out_dim:
        times = lambda x:  x
        trans = lambda x : x
    elif in_dim > out_dim:
        # we drop some samples
        times = lambda x:  x[:out_dim]
        # we pad with zeros
        trans = lambda x : jnp.pad(x, (0, in_dim - out_dim))
    else:
        # we pad with zeros
        times = lambda x : jnp.pad(x, (0, out_dim - in_dim))
        # we drop some samples
        trans = lambda x:  x[:in_dim]
    
    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(out_dim,in_dim))
