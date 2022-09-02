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
Software for Processing of Geophysical Signals
"""

# pylint: disable=W0611

from cr.sparse._src.geo.wavelets import (
    ricker
)

from cr.sparse._src.geo.thresholding import (
    # Thresholding operators
    hard_threshold,
    hard_threshold_jit,
    soft_threshold,
    soft_threshold_jit,
    half_threshold,
    half_threshold_jit,
    hard_threshold_percentile,
    hard_threshold_percentile_jit,
    soft_threshold_percentile,
    soft_threshold_percentile_jit,
    half_threshold_percentile,
    half_threshold_percentile_jit,
    gamma_to_tau_half_threshold,
    gamma_to_tau_hard_threshold,
)
