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

import jax
import jax.numpy as jnp
from jax import random, jit

from cr.nimble import promote_arg_dtypes
from cr.nimble import arr_l2norm_sqr

def awgn_at_snr(key, signal, snr):
    """Generates noise for the signal at the specified SNR
    """
    signal = jnp.asarray(signal)
    signal = promote_arg_dtypes(signal)
    n = signal.size
    energy = arr_l2norm_sqr(signal)
    mean_energy = energy / n
    mean_energy_db = 10*jnp.log10(mean_energy)
    noise_mean_energy_db = mean_energy_db - snr
    sigma = 10**(noise_mean_energy_db/20)
    noise = sigma * random.normal(key, signal.shape)
    return noise