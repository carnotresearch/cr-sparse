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
Metrics
"""
# pylint: disable=W0611


from cr.sparse._src.metrics import (
    mean_squared,
    mean_squared_error,
    root_mean_squared,
    root_mse,
    normalization_factor,
    normalized_root_mse,
    peak_signal_noise_ratio,
    signal_noise_ratio
)