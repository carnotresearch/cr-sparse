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
Spectral Clustering Based Algorithms
"""

from cr.sparse._src.cluster.spectral import (
    SpectralclusteringSolution,
    # laplacians
    normalized_random_walk_laplacian,
    normalized_random_walk_laplacian_jit,
    # spectral clustering algorithms
    # unnormalized spectral clustering
    unnormalized,
    unnormalized_k,
    unnormalized_k_jit,
    # normalized random walk clustering
    normalized_random_walk,
    normalized_random_walk_k,
    normalized_random_walk_k_jit    
)