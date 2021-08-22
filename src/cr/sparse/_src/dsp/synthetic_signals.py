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

import math
import jax.numpy as jnp

def time_values(fs, T, initial_time=0):
    # Number of samples
    n = int(fs * T)
    # Points in time where the chirp will be computed.
    t = jnp.linspace(initial_time, initial_time+T, n, endpoint=False)
    return t

def chirp(fs, T, f0, f1, initial_phase=0):
    """Generates a frequency sweep from low to high over time.

    Args:
        fs (float): Sample rate of signal in Hz.
        T (float): Period of the signal in seconds.
        f0 (float): Start (lower) frequency of chirp in Hz.
        f1 (float): Stop (upper) frequency of chirp in Hz.
        initial_phase (float): phase at t=0 in radians, default is 0.

    Returns:
        A tuple comprising (i) an array of time values in seconds and (ii) an array of signal values

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


def chirp_centered(fs, T, fc, bw, initial_phase=0):
    """Generates a frequency sweep from low to high over time defined by central frequency and bandwidth.

    Args:
        fs (float): Sample rate of signal in Hz.
        T (float): Period of the signal in seconds.
        fc (float): Central frequency of chirp in Hz.
        bw (float): Bandwidth (end frequency -  start frequency) of chirp in Hz.
        initial_phase (float): phase at t=0 in radians, default is 0.

    Returns:
        A tuple comprising (i) an array of time values in seconds and (ii) an array of signal values

    Adapted from https://udel.edu/~mm/gr/chirp.py
    """
    f0 = fc - bw / 2.
    f1 = fc + bw / 2.
    return chirp(fs, T, f0, f1, initial_phase)



def pulse(fs, T, start_time, end_time, initial_time=0):
    """Generates a pulse signal which is 1 between start and end times and 0 everwhere else

    Args:
        fs (float): Sample rate of signal in Hz.
        T (float): Period of the signal in seconds.
        start_time (float): Start time of the box signal in seconds
        end_time (float): End time of the box signal in seconds
        initial_time (float): time at waveform start in seconds, default is 0.

    Returns:
        A tuple comprising (i) an array of time values in seconds and (ii) an array of signal values
    """
    t = time_values(fs, T, initial_time)
    signal = jnp.zeros_like(t)
    index = jnp.logical_and(t >= start_time, t < end_time)
    signal = signal.at[index].set(1)
    return t, signal

def gaussian(fs, T, b, a=1., initial_time=0):
    """Generates a Gaussian signal

    Args:
        fs (float): Sample rate of signal in Hz.
        T (float): Period of the signal in seconds.
        b (float): The location (in time) where the pulse is centered in seconds.
        a (float): scale of Gaussian signal (in seconds).
        initial_time (float): time at waveform start in seconds, default is 0.

    Returns:
        A tuple comprising (i) an array of time values in seconds and (ii) an array of signal values

    """
    t = time_values(fs, T, initial_time)
    a = jnp.atleast_2d(jnp.asarray(a)).T
    tb = t - b
    # square the scale s^2
    wsq = a**2
    # t^2
    xsq = tb**2
    # the gaussian term e^{-t^2/2a^2}
    gauss = jnp.exp(-xsq / (2 * wsq))
    return t, jnp.squeeze(gauss)

def decaying_sine_wave(fs, T, f, alpha, initial_phase=0, initial_time=0):
    """Generates a decaying sinusoid
    Args:
        fs (float): Sample rate of signal in Hz.
        T (float): Period of the signal in seconds.
        f (float): Frequency of the sine wave in Hz.
        alpha (float): Exponential decay factor in Hz.
        initial_phase (float): phase at t=0 in radians, default is 0.
        initial_time (float): time at waveform start in seconds, default is 0.

    Returns:
        A tuple comprising (i) an array of time values in seconds and (ii) an array of signal values
    """
    t = time_values(fs, T, initial_time)
    phase = 2*jnp.pi*f*t
    phase += initial_phase
    decay = jnp.exp(-alpha*t)
    signal = decay * jnp.sin(phase) 
    return t, signal

def transient_sine_wave(fs, T, f, start_time, end_time, initial_phase=0, initial_time=0):
    """Generates a transient sinusoid between start and end times

    Args:
        fs (float): Sample rate of signal in Hz.
        T (float): Period of the signal in seconds.
        f (float): Frequency of the sine wave in Hz.
        start_time (float): Start time of the sine wave in seconds
        end_time (float): End time of the sine wave in seconds
        initial_phase (float): phase at t=0 in radians, default is 0.
        initial_time (float): time at waveform start in seconds, default is 0.

    Returns:
        A tuple comprising (i) an array of time values in seconds and (ii) an array of signal values

    Example:
        ::

            fs = 100
            T = 16
            f = 2
            start_time = 2
            end_time = 6
            initial_time = -4
            t, signal = transient_sine_wave(fs, T, f, start_time, end_time, initial_time=initial_time)        
    """
    t = time_values(fs, T, initial_time)
    phase = 2*jnp.pi*f*t
    phase += initial_phase
    signal = jnp.sin(phase)
    mask = jnp.logical_or(t < start_time, t >= end_time)
    signal = signal.at[mask].set(0)
    return t, signal

def gaussian_pulse(fs, T, b, fc=1000, bw=0.5, bwr=-6, retquad=False, retenv=False, initial_time=0):
    """Generates a Gaussian modulated sinusoid

    Args:
        fs (float): Sample rate of signal in Hz.
        T (float): Period of the signal in seconds.
        b (float): The location (in time) where the pulse is centered in seconds.
        fc (float): Center frequency of the Gaussian Pulse in Hz.
        bw (float): Fractional bandwidth in frequency domain of pulse.
        bwr (float): Reference level at which fractional bandwidth is calculated (dB).
        retquad (bool): Include/exclude quadrature(imaginary) part of the signal in the result
        retenv (bool): Include/exclude the Gaussian envelope of the signal in the result
        initial_time (float): time at waveform start in seconds, default is 0.

    Returns:
        tuple: A tuple comprising of 
        (i) t: time values in seconds at which the signal is computed.
        (ii) real_part: values of the real part of the signal at these time values.
        (iii) imag_part: values of the quadrature/imaginary part of the signal at these time values (if retquad is True).
        (iv) envelope: values of the Gaussian envelope of the signal at these time values (if retenv is True).

    Adapted from https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/waveforms.py#L161-L258
    """
    t = time_values(fs, T, initial_time)
    ref = math.pow(10.0, bwr / 20.0)
    a = -(jnp.pi * fc * bw) ** 2 / (4.0 * math.log(ref))
    tb = t - b
    envelope = jnp.exp(-a * tb * tb)
    real_part = envelope * jnp.cos(2 * jnp.pi * fc * tb)
    if retquad:
        imag_part = envelope * jnp.sin(2 * jnp.pi * fc * tb)
    if retquad:
        if retenv:
            return t, real_part, imag_part, envelope
        else:
            return t, real_part, imag_part
    else:
        if retenv:
            return t, real_part, envelope
        else:
            return t, real_part

