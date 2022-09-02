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
Wavelets popular in geophysics literature
"""

import jax.numpy as jnp

def ricker(t, f0=10):
    """
    Returns the Ricker wavelet for a given peak frequency at specified times.

    There are several different ways in which the Ricker wavelet can be 
    defined.

    Here we are interested in the definition popular in seismology.
    
    See https://subsurfwiki.org/wiki/Ricker_wavelet for details.
    """
    w = (1 - 2 * (jnp.pi * f0 * t) ** 2) * jnp.exp(-(jnp.pi * f0 * t) ** 2)
    return w
