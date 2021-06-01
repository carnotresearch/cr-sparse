# Copyright 2021 Carnot Research Pvt Ltd
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

from .norm import sqr_norms_l2_cw, norms_l2_cw

norm = jnp.linalg.norm


class SignalsComparison:

    def __init__(self, references, estimates):
        self.references = references
        self.estimates  = estimates
        self.differences = references - estimates

    @property
    def reference_norms(self):
        return norms_l2_cw(self.references)

    @property
    def estimate_norms(self):
        return norms_l2_cw(self.estimates)

    @property
    def difference_norms(self):
        return norms_l2_cw(self.differences)

    @property
    def reference_energies(self):
        return sqr_norms_l2_cw(self.references)

    @property
    def estimate_energies(self):
        return sqr_norms_l2_cw(self.estimates)

    @property
    def difference_energies(self):
        return sqr_norms_l2_cw(self.differences)

    @property
    def error_to_signal_norms(self):
        a_norms = self.reference_norms
        diff_norms = self.difference_norms
        return diff_norms / a_norms
    
    @property
    def signal_to_noise_ratios(self):
        a_energies = self.reference_energies
        diff_energies = self.difference_energies
        ratios = a_energies / diff_energies
        return 10*jnp.log10(ratios)

    @property
    def cum_reference_norm(self):
        return norm(self.references, 'fro')

    @property
    def cum_estimate_norm(self):
        return norm(self.estimates, 'fro')

    @property
    def cum_difference_norm(self):
        return norm(self.differences, 'fro')

    @property
    def cum_error_to_signal_norm(self):
        a_norm = self.cum_reference_norm
        err_norm = self.cum_difference_norm
        ratio = err_norm / a_norm
        return ratio

    @property
    def cum_signal_to_noise_ratio(self):
        a_norm = self.cum_reference_norm
        err_norm = self.cum_difference_norm
        ratio = a_norm  / err_norm
        return 20 * jnp.log10(ratio)

    def summarize(self):
        n, s = self.references.shape
        print(f'Dimensions: {n}')
        print(f'Signals: {s}')
        print(f'Combined reference norm: {self.cum_reference_norm:.3f}')
        print(f'Combined estimate norm: {self.cum_estimate_norm:.3f}')
        print(f'Combined difference norm: {self.cum_difference_norm:.3f}')
        print(f'Combined SNR: {self.cum_signal_to_noise_ratio:.2f} dB')

