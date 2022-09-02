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
Phase transitions
"""
import math

import jax.numpy as jnp

import cr.sparse as crs

from .defs import PTConfig, PTConfigurations

def configuration(N):
    # N is number of atoms in dictionary/sensing matrix
    assert crs.is_power_of_2(N) and N >= 32
    # Minimum number of measurements
    M_min = 4
    if N <= 256:
        M_min = 16
    else:
        M_min = 32
    # Minimum undersampling ratio
    eta_min = M_min / N
    # Number of possible values of K for each value of M, sparsity levels
    num_rhos = min(32, N //4)
    # K/M s 
    rhos = 0.5 * jnp.arange(1, num_rhos+1) / num_rhos
    # Number of undersampling ratios (up to a max of M = N // 2)
    num_etas = N // (2 * M_min)
    # Undersampling levels
    Ms = jnp.arange(1, num_etas+1) * M_min
    # Undersampling ratios
    etas = Ms / N
    #print(rhos)
    #print(etas)
    print(Ms)
    added = set()
    configurations = []
    reverse_map = {}
    for m in range(num_etas):
        eta = float(etas[m])
        M = int(Ms[m])
        for k in range(num_rhos):
            rho = float(rhos[k])
            K = math.ceil(rho*M)
            reverse_map[(rho, eta)] = (K, M)
            if (K, M) in added:
                continue
            config = PTConfig(K=K, M=M, eta=eta, rho=rho)
            added.add((K,M))
            configurations.append(config)
    result = PTConfigurations(N=N, 
        configurations=configurations, 
        Ms=Ms, etas=etas, rhos=rhos,
        reverse_map=reverse_map)
    return result
    