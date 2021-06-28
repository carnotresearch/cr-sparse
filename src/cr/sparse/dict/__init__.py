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
Sparsifying dictionaries
"""

from cr.sparse._src.dict.simple import (
    gaussian_mtx,
    rademacher_mtx,
    random_onb,
    hadamard,
    hadamard_basis,
    dirac_hadamard_basis,
    dct_basis,
    dirac_dct_basis,
    dirac_hadamard_dct_basis,
    fourier_basis,
    random_orthonormal_rows
)

from cr.sparse._src.dict.props import (
    gram,
    frame,
    coherence_with_index,
    coherence,
    frame_bounds,
    upper_frame_bound,
    lower_frame_bound,
    babel,
    mutual_coherence_with_index,
    mutual_coherence,
)

from cr.sparse._src.dict.comparison import (
    matching_atoms_ratio
)