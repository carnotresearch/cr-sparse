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

"""
Some basic linear transformations
"""

import jax
import jax.numpy as jnp

from cr.sparse import promote_arg_dtypes

def point2d(x,y):
    """A point in 2D vector space"""
    return jnp.array([x+0.,y])

def vec2d(x,y):
    """A vector in 2D vector space"""
    return jnp.array([x+0.,y])

def rotate2d_cw(theta):
    """Construct an operator that rotates a 2D vector by angle :math:`\theta` clock-wise
    """
    Q  = jnp.array([[jnp.cos(theta), jnp.sin(theta)],
        [-jnp.sin(theta), jnp.cos(theta)]])
    return Q

def rotate2d_ccw(theta):
    """Construct an operator that rotates a 2D vector by angle :math:`\theta` counter-clock-wise
    """
    Q  = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]])
    return Q

def reflect2d(theta):
    """Construct an operator that reflects a 2D vector across a line defined at angle :math:`\theta/2`
    """
    R  = jnp.array([[jnp.cos(theta), jnp.sin(theta)],
        [jnp.sin(theta), -jnp.cos(theta)]])
    return R
