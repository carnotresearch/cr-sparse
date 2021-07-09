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
norm = jnp.linalg.norm


from cr.sparse import norms_l2_cw

def project_to_ball(x, radius=1.0):
    x_norm = norm(x)
    factor = radius / x_norm
    return jnp.where(x_norm > radius, factor * x, x)


def project_to_box(x, radius=1.0):
    abs_x = jnp.abs(x)
    factors = jnp.maximum(abs_x, radius)
    factors = radius / factors
    return x * factors


def project_to_real_upper_limit(x, limit=1.0):
    return jnp.minimum(jnp.real(x), limit)
