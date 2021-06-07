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


import jax.numpy as jnp

from cr.sparse import (nonzero_indices, 
    largest_indices, 
    build_signal_from_indices_and_values
)

class RecoveryPerformance:
    """Performance of a sparse signal recovery operation
 
    * Synthesis :math:`y = \Phi x + e`
    * Recovery: :math:`y = \Phi \hat{x} + r`
    * Representation error: :math:`h = x - \hat{x}`
    * Residual: :math:`y - \Phi \hat{x} `
    """
    def __init__(self, Phi, y, x, x_hat=None, sol=None):
        """Computes all parameters related to the quality of reconstruction
        """
        # Shape of the dictionary/sensing matrix
        M, N = Phi.shape
        if sol is not None:
            x_hat = build_signal_from_indices_and_values(N, sol.I, sol.x_I) 
        # The K non-zero coefficients in x (set of indices)
        self.T0 = nonzero_indices(x)
        K = self.T0.size
        self.M = M
        self.N = N
        self.K = K
        # Norm of representation
        self.x_norm = jnp.linalg.norm(x)
        # Norm of measurement/signal
        self.y_norm = jnp.linalg.norm(y)
        # Norm of the reconstructed representation
        self.x_hat_norm = jnp.linalg.norm(x_hat)
        # recovery error vector. N length vector
        h = x - x_hat
        self.h = h
        # l_2 norm of representation error
        self.h_norm = jnp.linalg.norm(h)
        # recovery SNR
        self.recovery_snr = 20 * jnp.log10(self.x_norm / self.h_norm)
        # The portion of recovery error over T0 K length vector
        self.h_T0 = h[self.T0] 
        # Positions of other places (set of indices)
        index_set = jnp.arange(N)
        self.T0C = jnp.setdiff1d(index_set , self.T0)
        # Recovery error at T0C places N length vector
        hT0C = h.at[self.T0].set(0)
        self.h_T0C = hT0C
        # The K largest indices after T0 in recovery error (set of indices)
        self.T1 = largest_indices(hT0C, K)
        # The recovery error component over T1. [K] length vector.
        self.h_T1 = h[self.T1]
        # Remaining indices [N - 2K] set of indices
        self.TRest = jnp.setdiff1d(self.T0C , self.T1)
        # Recovery error over remaining indices [N - 2K] length vector
        self.h_TRest = h[self.TRest]
        # largest indices of the recovered vector
        self.R0 = jnp.sort(largest_indices(x_hat, K))
        # Support Overlap
        self.overlap = jnp.intersect1d(self.T0, self.R0)
        self.num_correct_atoms = self.overlap.size
        # Support recovery ratio
        self.support_recovery_ratio = self.num_correct_atoms / K
        # measurement/signal residual vector [M] length vector
        r = y - Phi @ x_hat
        self.residual = r
        # Norm of measurement error.  This must be less than epsilon
        self.r_norm = jnp.linalg.norm(r)
        # Measurement SNR
        self.measurement_snr = 20 * jnp.log10(self.y_norm / self.r_norm)
        # Ratio between the norm of recovery error and measurement error
        self.h_by_r_norms = self.h_norm / self.r_norm
        # Whether we consider the process to be success or not.
        # We consider success only if the support has been recovered
        # completely.
        self.success = self.num_correct_atoms >= K

    def print(self):
        print(f'M: {self.M}, N: {self.N}, K: {self.K}')
        print(f'x_norm: {self.x_norm:.3f}, y_norm: {self.y_norm:.3f}')
        print(f'x_hat_norm: {self.x_hat_norm:.3f}, h_norm: {self.h_norm:.2e}, r_norm: {self.r_norm:.2e}')
        print(f'recovery_snr: {self.recovery_snr:.2f} dB, measurement_snr: {self.measurement_snr:.2f} dB')
        print(f'T0: {self.T0}')
        print(f'R0: {self.R0}')
        print(f'Overlap: {self.overlap}, Correct: {self.num_correct_atoms}')
        print(f'success: {self.success}')
