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
Thresholding operators as defined in the paper: 

.. [1] Chen, Y., Chen, K., Shi, P., Wang, Y., “Irregular seismic
       data reconstruction using a percentile-half-thresholding algorithm”,
       Journal of Geophysics and Engineering, vol. 11. 2014.       
"""

import numpy as np
from jax import jit
import jax.numpy as jnp

from cr.nimble import promote_arg_dtypes

def hard_threshold(x, tau):
    """Hard threshold
    """
    x = promote_arg_dtypes(x)
    # The terms that will remain non-zero after thresholding
    gamma = jnp.sqrt(2*tau)
    nonzero = jnp.abs(x) > gamma
    return nonzero * x

hard_threshold_jit = jit(hard_threshold, static_argnums=(1,))

def soft_threshold(x, tau):
    """Soft threshold
    """
    x = promote_arg_dtypes(x)
    if np.iscomplexobj(x):
        return jnp.maximum(jnp.abs(x) - tau, 0.) * jnp.exp(1j * jnp.angle(x))
    else:
        return jnp.maximum(0, x - tau) + jnp.minimum(0, x + tau)

soft_threshold_jit = jit(soft_threshold, static_argnums=(1,))

def half_threshold(x, tau):
    r"""Half threshold
    """
    x = promote_arg_dtypes(x)
    gamma = (54 ** (1. / 3.) / 4.) * tau ** (2. / 3.)
    nonzero = jnp.abs(x) >= gamma
    # the arc-cos term Eq 10 from paper
    phi = 2. / 3. * jnp.arccos((tau / 8.) * (jnp.abs(x) / 3.) ** (-1.5))
    # the half thresholded values for terms above gamma Eq 10
    x = 2./3. * x * (1 + jnp.cos(2. * jnp.pi / 3. - phi))
    # combine zero and non-zero terms
    return jnp.where(nonzero, x, jnp.zeros_like(x))

half_threshold_jit = jit(half_threshold, static_argnums=(1,))

def hard_threshold_percentile(x, perc):
    """Percentile hard threshold
    """
    x = promote_arg_dtypes(x)
    # desired gamma
    gamma = jnp.percentile(jnp.abs(x), perc)
    # convert gamma to tau
    tau = 0.5 * gamma ** 2
    return hard_threshold(x, tau)

hard_threshold_percentile_jit = jit(hard_threshold_percentile, static_argnums=(1,))

def soft_threshold_percentile(x, perc):
    """Percentile soft threshold
    """
    x = promote_arg_dtypes(x)
    # desired gamma and tau are same
    tau = jnp.percentile(jnp.abs(x), perc)
    return soft_threshold(x, tau)

soft_threshold_percentile_jit = jit(soft_threshold_percentile, static_argnums=(1,))

def half_threshold_percentile(x, perc):
    """Percentile half threshold
    """
    x = promote_arg_dtypes(x)
    gamma = jnp.percentile(jnp.abs(x), perc)
    # convert gamma to tau
    tau = (4. / 54 ** (1. / 3.) * gamma) ** 1.5
    return half_threshold(x, tau)

half_threshold_percentile_jit = jit(half_threshold_percentile, static_argnums=(1,))

def gamma_to_tau_half_threshold(gamma):
    """Converts gamma to tau for half thresholding
    """
    return (4. / 54 ** (1. / 3.) * gamma) ** 1.5

def gamma_to_tau_hard_threshold(gamma):
    """Converts gamma to tau for hard thresholding
    """
    return  0.5 * gamma ** 2
