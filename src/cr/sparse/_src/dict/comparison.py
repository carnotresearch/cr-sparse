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


def matching_atoms_ratio(A, B, distance_threshold=0.01):
    """Identifies how many atoms are very close between dictionaries A and B
    """
    # number of atoms
    n_atoms = A.shape[1]
    # inner products betweeen atoms of A and B
    similarities = jnp.abs(A.T @ B)
    # find the best match from each row 
    max_similarities = jnp.max(similarities, axis=1)
    # angular distance between best match
    min_distances = 1 - max_similarities
    # print(min_distances)
    # count how many matches are below the distance threshold
    matches = min_distances < distance_threshold
    count = sum(matches)
    # print(count)
    # ratio of matches
    return count / n_atoms
