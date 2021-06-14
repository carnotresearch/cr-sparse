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

from .util import promote_arg_dtypes

def is_scalar(x):
    return x.ndim == 0

def is_vec(x):
    return x.ndim == 1 or (x.ndim == 2 and 
        (x.shape[0] == 1 or x.shape[1] == 1))

def is_line_vec(x):
    return x.ndim == 1

def is_row_vec(x):
    return x.ndim == 2 and x.shape[0] == 1 

def is_col_vec(x):
    return x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)

def to_row_vec(x):
    assert x.ndim == 1
    return jnp.expand_dims(x, 0)

def to_col_vec(x):
    assert x.ndim == 1
    return jnp.expand_dims(x, 1)