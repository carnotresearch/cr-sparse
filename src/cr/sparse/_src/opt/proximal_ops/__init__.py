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

def prox_value_vec(func, prox_op):
    """Returns a function which computes both proximal value and vector
    """
    @partial(jit, static_argnums=(2,))
    def operator(x, t, mode=0):
        """
        mode=0 only function value at x
        mode=1 only proximal vector for x
        mode=2 both proximal vector for x and function value at the proximal vector
        """
        if mode == 0:
            return func(x)
        if mode == 1:
            return prox_op(x, t)
        if mode == 2:
            x = prox_op(x, t)
            v = func(x)
            return v, x
        return None

    return operator