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

def chirp(fs, f0, f1, T, initial_phase=0):
    """Generates a frequency sweep from low to high over time.
    
    Args:
        fs (float): Sample rate of chirp signal in Hz.
        f0 (float): Start (lower) frequency of chirp in Hz.
        f1 (float): Stop (upper) frequency of chirp in Hz.
        T (float): Period of the chirp in seconds.
        initial_phase (float): , phase at waveform start in radians, default is 0.

    Returns:
        jax.numpy.ndarray: Time domain chirp waveform.

    Adapted from https://udel.edu/~mm/gr/chirp.py
    """
    # Chirp rate in Hz/s.
    c = (f1 - f0) / T 
    # Number of samples
    n = int(fs * T)
    # Points in time where the chirp will be computed.
    t = jnp.linspace(0, T, n, endpoint=False)
    # Instantaneous phase in Hz is integral of frequency, f(t) = ct + f0.
    phase_hz = (c * t**2) / 2 + (f0 * t)
    # Convert to radians.
    phase_rad = 2 * jnp.pi * phase_hz 
    # Offset by user-specified initial phase
    phase_rad += initial_phase
    # compute the chirp signal at the specified points in time
    signal = jnp.cos(phase_rad)
    return t, signal
