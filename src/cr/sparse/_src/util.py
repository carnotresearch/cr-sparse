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
    """Promotes `args` to a common inexact type."""
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
    if dtype is None:
        return dtype
    return dtypes.canonicalize_dtype(dtype)