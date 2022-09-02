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
Data Clustering Algorithms
"""

from cr.sparse._src.cluster.util import (
    sizes_from_labels,
    sizes_from_labels_jit,
    start_end_indices,
    labels_from_sizes,
    best_map,
    best_map_k,
    ClusteringError,
    clustering_error,
    clustering_error_k
)
