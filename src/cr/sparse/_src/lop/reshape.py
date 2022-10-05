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

from functools import reduce
import jax.numpy as jnp

from .lop import Operator


def reshape(in_shape, out_shape, order='C'):
    """Returns a linear operator which reshapes vectors from model space to data space

    Args:
        in_shape (int): Shape of vectors in the model space 
        out_shape (int): Shape of vectors in the data space
        order: Specifies index order of data layout ['C', 'F', 'A']
            C means C-like index order (default). F means Fortran
            like order. This is the order in MATLAB arrays also.

    Returns:
        (Operator): A reshaping linear operator
    """
    in_size = jnp.prod(jnp.array(in_shape))
    out_size = jnp.prod(jnp.array(out_shape))
    assert in_size == out_size, "Input and output size must be equal"
    assert order in ['C', 'F', 'A'], "Invalid order"

    times = lambda x:  jnp.reshape(x, out_shape, order=order)
    trans = lambda x : jnp.reshape(x, in_shape, order=order)
    return Operator(times=times, trans=trans, shape=(out_shape,in_shape))


def arr2vec(shape):
    """Returns a linear operator which reshapes arrays to vectors

    Args:
        shape (int): Shape of arrays in the model space 

    Returns:
        (Operator): An array to vec linear operator
    """
    in_size = reduce((lambda x, y: x * y), shape)
    out_shape = (in_size,)

    times = lambda x:  jnp.reshape(x, (in_size,))
    trans = lambda x : jnp.reshape(x, shape)
    return Operator(times=times, trans=trans, shape=(out_shape,shape))
