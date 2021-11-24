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
    build_signal_from_indices_and_values,
    dynamic_range,
    nonzero_dynamic_range
)

class RecoveryPerformance:
    """Performance of a sparse signal recovery operation
 
    * Synthesis :math:`y = \\Phi x + e`
    * Recovery: :math:`y = \\Phi \\hat{x} + r`
    * Representation error: :math:`h = x - \\hat{x}`
    * Residual: :math:`y - \\Phi \\hat{x}`
    """

    M : int = 0
    """Signal/Measurement space dimension, number of rows in :math:`\\Phi`"""
    N : int = 0
    """Representation space dimension, number of atoms/columns in :math:`\\Phi`"""
    K: int = 0
    """Number of non-zero entries in :math:`x`"""
    T0 = []
    """The index set of K non-zero coefficients in :math:`x`"""
    x_norm: float = 0
    """norm of representation :math:`x`"""
    y_norm: float = 0
    """norm of measurement/signal :math:`y`"""
    x_hat_norm: float = 0
    """norm of the reconstruction :math:`\\hat{x}`"""
    x_dr: float = 0
    """ Dynamic range of x """
    y_dr: float = 0
    """ Dynamic range of y """
    x_hat_dr: float = 0
    """ Dynamic range of x_hat """
    h = []
    """Recovery/reconstruction error :math:`h = x - \\hat{x}`"""
    h_norm: float = 0
    """Norm of reconstruction error :math:`h`"""
    recovery_snr: float = 0
    """Reconstruction/recovery SNR (dB) in representation space :math:`20 \\log (\\| x \\|_2 / \\| h \\|_2)`"""
    R0 = []
    """Index set of K largest (magnitude) entries in the reconstruction :math:`\\hat{x}`"""
    overlap = []
    """Indices overlapping between T0 and R0  :math:`T_0 \\cap R_0`"""
    num_correct_atoms : int = 0
    """Number of entries in the overlap, i.e. number of indices of the support correctly recovered"""
    r = []
    """The residual :math:`r = y - \\Phi \\hat{x}`"""
    r_norm : float = 0
    """Norm of the residual"""
    measurement_snr: float = 0
    """Measurement SNR (dB) in measurement/signal space :math:`20 \\log (\\| y \\|_2 / \\| r \\|_2)`"""

    def __init__(self, Phi, y, x, x_hat=None, sol=None):
        """Computes all parameters related to the quality of reconstruction
        """
        # Shape of the dictionary/sensing matrix
        M, N = Phi.shape
        if sol is not None:
            x_hat = build_signal_from_indices_and_values(N, sol.I, sol.x_I) 
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
        self.x_dr = nonzero_dynamic_range(x)
        self.y_dr = dynamic_range(y)
        self.x_hat_dr = nonzero_dynamic_range(x_hat)
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
        # measurement/signal residual vector [M] length vector
        r = y - (Phi.times(x_hat) if hasattr(Phi, 'times') else Phi @ x_hat)
        self.r = r
        # Norm of measurement error.  This must be less than epsilon
        self.r_norm = jnp.linalg.norm(r)
        # Measurement SNR
        self.measurement_snr = 20 * jnp.log10(self.y_norm / self.r_norm)
        # Ratio between the norm of recovery error and measurement error
        self.h_by_r_norms = self.h_norm / self.r_norm

    def print(self):
        """Prints metrics related to reconstruction quality"""
        print(f'M: {self.M}, N: {self.N}, K: {self.K}')
        print(f'x_norm: {self.x_norm:.3f}, y_norm: {self.y_norm:.3f}')
        print(f'x_hat_norm: {self.x_hat_norm:.3f}, h_norm: {self.h_norm:.2e}, r_norm: {self.r_norm:.2e}')
        print(f'recovery_snr: {self.recovery_snr:.2f} dB, measurement_snr: {self.measurement_snr:.2f} dB')
        print(f'x_dr: {self.x_dr:.2f} dB, y_dr: {self.y_dr:.2f} dB, x_hat_dr: {self.x_hat_dr:.3f} dB')
        print(f'T0: {self.T0}')
        print(f'R0: {self.R0}')
        print(f'Overlap: {self.overlap}')
        print(f'Correct atoms: {self.num_correct_atoms}. Ratio: {self.support_recovery_ratio:.2f}, perfect_support_recovery: {self.perfect_support_recovery}')
        print(f'success: {self.success}')

    @property
    def support_recovery_ratio(self):
        """Returns the ratio of correctly recovered atoms"""
        return self.num_correct_atoms / self.K

    @property
    def perfect_support_recovery(self):
        """Returns if the support has been recovered perfectly"""
        return self.num_correct_atoms >= self.K

    @property
    def success(self):
        """Returns True if more than 75% indices are correctly identified and recovery SNR is high (> 30 dB)"""
        return bool(self.support_recovery_ratio > 0.75) and bool(self.recovery_snr > 30)

