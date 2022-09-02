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
Sparse Subspace Clustering Algorithms
"""

from cr.sparse._src.cluster.ssc.util import (
    sparse_to_full_rep,
    sparse_to_bcoo,
    bcoo_to_sparse,
    bcoo_to_sparse_jit,
    rep_to_affinity,
    angles_between_points,
    min_angles_inside_cluster,
    min_angles_outside_cluster,
    nearest_neighbors_inside_cluster,
    nearest_neighbors_outside_cluster,
    sorted_neighbors,
    inn_positions,
    subspace_preservation_stats,
    subspace_preservation_stats_jit,
    sparse_subspace_preservation_stats,
    sparse_subspace_preservation_stats_jit,
)


from cr.sparse._src.cluster.ssc.omp import (
    build_representation_omp,
    build_representation_omp_jit,
    batch_build_representation_omp,
    batch_build_representation_omp_jit,
)