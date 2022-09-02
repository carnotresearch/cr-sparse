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

from functools import partial
from jax import jit

import jax.numpy as jnp
import cr.nimble as cnb

def smooth_value_grad(func, grad):
    """Returns a function which computes both the value and gradient of a smooth function at a specified point
    """
    @partial(jit, static_argnums=(1,))
    def evaluator(x, mode=0):
        """
        mode=0 only value
        mode=1 only gradient
        mode=2 both value and gradient
        """
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        if mode == 0:
            return func(x)
        if mode == 1:
            return grad(x)
        if mode == 2:
            v = func(x)
            g = grad(x)
            return v, g
        return None

    return evaluator