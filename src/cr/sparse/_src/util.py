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


import jax.numpy as jnp
from jax import lax
from jax._src import dtypes



def promote_arg_dtypes(*args):
    """Promotes `args` to a common inexact type.
    
    Args:
        *args: list of JAX ndarrays to be promoted to common inexact type

    Returns:
        The same list of arrays with their dtype promoted to a common inexact type

    Example:
        Promoting a single argument::

            >>> cr.sparse.promote_arg_dtypes(jnp.arange(2))
            DeviceArray([0., 1.], dtype=float32)
            >>> from jax.config import config
            >>> config.update("jax_enable_x64", True)
            >>> cr.sparse.promote_arg_dtypes(jnp.arange(2))
            DeviceArray([0., 1.], dtype=float64)

        Promoting two arguments to common floating point type::

            >>> a = jnp.arange(2)
            >>> b = jnp.arange(1.5, 3.5)
            >>> a, b = cr.sparse.promote_arg_dtypes(a, b)
            >>> print(a)
            >>> print(b)
            [0. 1.]
            [1.5 2.5]

        A mix of real and complex types::

            >>> a = jnp.arange(2) + 0.j
            >>> b = jnp.arange(1.5, 3.5)
            >>> a, b = cr.sparse.promote_arg_dtypes(a, b)
            >>> print(a)
            >>> print(b)
            [0.+0.j 1.+0.j]
            [1.5+0.j 2.5+0.j]
    """
    def _to_inexact_type(type):
        return type if jnp.issubdtype(type, jnp.inexact) else jnp.float_
    inexact_types = [_to_inexact_type(arg.dtype) for arg in args]
    dtype = dtypes.canonicalize_dtype(jnp.result_type(*inexact_types))
    args = [lax.convert_element_type(arg, dtype) for arg in args]
    if len(args) == 1:
        return args[0]
    else:
        return args


def canonicalize_dtype(dtype):
    """Wrapper function on dtypes.canonicalize_dtype with None handling
    """
    if dtype is None:
        return dtype
    return dtypes.canonicalize_dtype(dtype)

def check_shapes_are_equal(array1, array2):
    """Raise an error if the shapes of the two arrays do not match.
    
    Raises:
        ValueError: if the shape of two arrays is not same
    """
    if not array1.shape == array1.shape:
        raise ValueError('Input arrays must have the same shape.')
    return


integer_types = (
    jnp.uint8.dtype,
    jnp.uint16.dtype,
    jnp.uint32.dtype,
    jnp.uint64.dtype,
    jnp.int8.dtype,
    jnp.int16.dtype,
    jnp.int32.dtype,
    jnp.int64.dtype,
)

integer_ranges = {t: (jnp.iinfo(t).min, jnp.iinfo(t).max)
                   for t in integer_types}


dtype_ranges = {
    bool: (False, True),
    float: (-1, 1),
    jnp.bool_.dtype: (False, True),
    jnp.float_.dtype: (-1, 1),
    jnp.float16.dtype: (-1, 1),
    jnp.float32.dtype: (-1, 1),
    jnp.complex64.dtype: (-1, 1),
    jnp.complex128.dtype: (-1, 1),
}

dtype_ranges.update(integer_ranges)
