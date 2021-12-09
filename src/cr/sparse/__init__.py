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
CR-Sparse
"""
# pylint: disable=W0611


from .version import __version__


from cr.sparse._src.discrete.number import (
    next_pow_of_2,
    is_integer,
    is_positive_integer,
    is_negative_integer,
    is_odd,
    is_even,
    is_odd_natural,
    is_even_natural,
    is_power_of_2,
    is_perfect_square,
    integer_factors_close_to_sqr_root
)

from cr.sparse._src.signal import (
    nonzero_values,
    nonzero_indices,
    support,
    randomize_rows,
    randomize_cols,
    largest_indices,
    hard_threshold,
    hard_threshold_sorted,
    sparse_approximation,
    build_signal_from_indices_and_values,
    hard_threshold_by,
    largest_indices_by,
    dynamic_range,
    nonzero_dynamic_range,

    # row wise
    largest_indices_rw,
    take_along_rows,
    sparse_approximation_rw,

    # column wise
    largest_indices_cw,
    take_along_cols,
    sparse_approximation_cw,

    # energy of a signal
    energy,

    # statistical normalization of data
    normalize,
    normalize_jit,

    # interpolate via fourier transform
    interpft,

    # convolution
    vec_convolve,
    vec_convolve_jit,
)

from cr.sparse._src.signal import (

    find_first_signal_with_energy_le_rw,
    find_first_signal_with_energy_le_cw,
)

from cr.sparse._src.signal import (
    frequency_spectrum,
    power_spectrum
)


from cr.sparse._src.signalcomparison import (
    SignalsComparison,
    snrs_cw,
    snrs_rw,
    snr
)

from cr.sparse._src.noise import (
    awgn_at_snr
)

from cr.sparse._src.types import (
    RecoveryFullSolution
)


from cr.sparse._src.similarity import (
    dist_to_gaussian_sim,
    sqr_dist_to_gaussian_sim,
    eps_neighborhood_sim
)


# Evaluation Tools

from cr.sparse._src.tools.performance import (
    RecoveryPerformance
)

from cr.sparse._src.tools.trials_at_fixed_m_n import (
    RecoveryTrialsAtFixed_M_N
)
