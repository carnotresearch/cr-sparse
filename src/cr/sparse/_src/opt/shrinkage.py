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


def shrink(a, kappa):
    """Shrinks each entry of a vector by :math:`\\kappa`.

    Entries with magnitude below :math:`\\kappa` go to 0. 
    The magnitude of other entries is reduced by :math:`\\kappa`.
    """
    return jnp.maximum(0, a - kappa) + jnp.minimum(0, a + kappa)