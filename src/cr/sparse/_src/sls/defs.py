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

"""
Common definitions for sparse linear systems
"""
import numpy as np
from typing import NamedTuple, Callable
import jax.numpy as jnp

identity_func = lambda x : x

class SimpleOp(NamedTuple):
    times : Callable[[jnp.ndarray], jnp.ndarray]
    trans : Callable[[jnp.ndarray], jnp.ndarray]

identity_op = SimpleOp(times=identity_func, trans=identity_func)

def default_threshold(i, x):
    """Default thresholding function (no decay)
    """
    # Get the 20 percentile in magnitude
    tau = jnp.percentile(jnp.abs(x), 20)
    if np.iscomplexobj(x):
        return jnp.maximum(jnp.abs(x) - tau, 0.) * jnp.exp(1j * jnp.angle(x))
    else:
        return jnp.maximum(0, x - tau) + jnp.minimum(0, x + tau)
