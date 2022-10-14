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


import cr.nimble as crn
from matplotlib import pyplot as plt

def noop_tracker(state):
    pass


def norm_tracker(state):
    x_norm = crn.arr_l2norm(state.x)
    r_norm = crn.arr_l2norm(state.r)
    print(f'[{state.iterations}] x_norm: {x_norm:.3e}, r_norm: {r_norm:.3e}')

def print_tracker(state):
    print(state)


class ProgressTracker:
    """
    Progress tracker for sparse recovery algorithms
    """

    def __init__(self, x0=None):
        self._r_norms = []
        self._x_norms = []
        self._iterations = []
        self.x0 = x0
        self._snr = []

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

    def __call__(self, state):
        x = state.x
        r = state.r
        x_norm = crn.arr_l2norm(x)
        r_norm = crn.arr_l2norm(r)
        iterations = state.iterations
        self._r_norms.append(r_norm)
        self._x_norms.append(x_norm)
        self._iterations.append(iterations)
        msg = ''
        if self.x0 is not None:
            snr = crn.signal_noise_ratio(self.x0, x)
            self._snr.append(snr)
            msg = f', SNR: {snr:.2f} dB'
        print(f'[{iterations}] x_norm: {x_norm:.2e}, r_norm: {r_norm:.2e}{msg}')

    def plot_progress(self):
        fig, ax1 = plt.subplots(figsize=(15,5))
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
