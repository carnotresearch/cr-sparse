# Copyright 2022 CR-Suite Development Team
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

import math
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

def get_x_norm(state):
    if hasattr(state, 'x_I'):
        return norm(state.x_I)
    if hasattr(state, 'x'):
        return norm(state.x)
    return 0.

def get_r_norm(state):
    if hasattr(state, 'r_norm_sqr'):
        return math.sqrt(state.r_norm_sqr)
    if hasattr(state, 'r'):
        return norm(state.r)

def build_sparse_signal(length, indices, values):
    x = np.zeros(length, dtype=values.dtype)
    x[indices] = values
    return x


def get_x_like(state, x0):
    if hasattr(state, 'x'):
        return state.x
    if hasattr(state, 'x_I'):
        n = x0.size
        return build_sparse_signal(n, state.I, state.x_I)
    return np.zeros_like(x0)


def signal_noise_ratio(reference_arr, test_arr):
    ref_energy = np.abs(np.vdot(reference_arr, reference_arr))
    error = reference_arr - test_arr
    err_energy = np.abs(np.vdot(error, error))
    eps = np.finfo(float).eps
    # make sure that error energy is non-zero
    err_energy = err_energy if err_energy else eps
    # make sure that ref energy is non-zero
    ref_energy = ref_energy if ref_energy else eps
    return 10 * np.log10(ref_energy/ err_energy)


def noop_tracker(state, more=False):
    pass


def norm_tracker(state, more=False):
    x_norm = get_x_norm(state)
    r_norm = get_r_norm(state)
    print(f'[{state.iterations}] x_norm: {x_norm:.2e}, r_norm: {r_norm:.2e}')
    if not more:
        print(f'Algorithm converged in {state.iterations} iterations.')

def print_tracker(state, more=False):
    print(state)
    if not more:
        print(f'Algorithm converged in {state.iterations} iterations.')


class ProgressTracker:
    """
    Progress tracker for sparse recovery algorithms
    """

    def __init__(self, x0=None, every=1):
        self._r_norms = []
        self._x_norms = []
        self._iterations = []
        self.x0 = x0
        self._snr = []
        self._every = every

    @property
    def x_norms(self):
        return self._x_norms

    @property
    def r_norms(self):
        return self._r_norms

    @property
    def iterations(self):
        return self._iterations

    @property
    def snrs(self):
        return self._snr

    def __call__(self, state, more=False):
        x_norm = get_x_norm(state)
        r_norm = get_r_norm(state)
        iterations = state.iterations
        self._r_norms.append(r_norm)
        self._x_norms.append(x_norm)
        self._iterations.append(iterations)
        msg = ''
        x0 = self.x0
        if x0 is not None:
            snr = signal_noise_ratio(x0, get_x_like(state, x0))
            self._snr.append(snr)
            msg = f', SNR: {snr:.2f} dB'
        if iterations % self._every == 0 or not more:
            print(f'[{iterations}] x_norm: {x_norm:.2e}, r_norm: {r_norm:.2e}{msg}')
        if not more:
            print(f'Algorithm converged in {state.iterations} iterations.')

    def plot_progress(self, ax1):
        ax1.plot(self.iterations, self.r_norms, label='residual norm')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Residual norm')
        ax1.legend()
        ax1.grid()
        if self.x0 is not None:
            ax2 = ax1.twinx()
            ax2.plot(self.iterations, self.snrs, '-r', label='SNR')
            ax2.set_ylabel('SNR (dB)')
            ax2.legend()
