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

from jax import jit

import jax.numpy as jnp
import cr.sparse as crs

def smooth_value_grad(func, grad):
    """Returns a function which computes both the value and gradient of a smooth function at a specified point
    """
    @jit
    def evaluator(x):
        x = jnp.asarray(x)
        x = crs.promote_arg_dtypes(x)
        v = func(x)
        g = grad(x)
        return v, g

    return evaluator