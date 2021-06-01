# Copyright 2021 Carnot Research Pvt Ltd
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

from typing import NamedTuple, List, Dict
from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

@dataclass
class SingleRecoverySolution:
    signals: jnp.DeviceArray = None
    representations : jnp.DeviceArray = None
    residuals : jnp.DeviceArray =  None
    residual_norms : jnp.DeviceArray = None
    iterations: int = None
    support : jnp.DeviceArray = None

class RecoverySolution(NamedTuple):
    # The non-zero values
    x_I: jnp.DeviceArray
    # The support for non-zero values
    I: jnp.DeviceArray
    # The residual
    r: jnp.DeviceArray
    # The residual norm squared
    r_norm_sqr: jnp.DeviceArray
    # The number of iterations it took to complete
    iterations: int


class PTConfig(NamedTuple):
    K: int
    M: int
    eta: int
    rho: int
    
class PTConfigurations(NamedTuple):
    N: int
    configurations: List[PTConfig]
    Ms: jnp.DeviceArray
    etas: jnp.DeviceArray
    rhos: jnp.DeviceArray
    reverse_map: Dict
