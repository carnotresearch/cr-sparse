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
from functools import partial
from typing import NamedTuple, Callable, Tuple

from jax import jit, lax
import jax.numpy as jnp
import jax.numpy.fft as jfft

########################################################################################################
# Tuple Describing a Continuous Wavelet
########################################################################################################

class WaveletFunctions(NamedTuple):
    """Functions associated with the wavelet
    """
    is_complex: bool
    """Indicates if the wavelet is complex"""
    time : Callable[[jnp.ndarray, float], jnp.ndarray]
    """Returns the wavelet function in time domain at specified time points"""
    frequency : Callable[[jnp.ndarray, float], jnp.ndarray]
    """Returns the wavelet function in frequency domain at specified angular frequencies"""
    fourier_period: Callable[[float], float]
    """Returns the equivalent Fourier period of the wavelet at a particular scale"""
    scale_from_period: Callable[[float], float]
    """Returns the equivalent scale of the wavelet at a particular Fourier period"""
    coi: Callable[[float], float]
    """Returns the cone of influence for the CWT at a particular scale"""

    def fourier_frequency(self, scale):
        """
        Return the equivalent frequencies .
        This is equivalent to 1.0 / self.fourier_period
        """
        period = self.fourier_period(scale)
        return jnp.reciprocal(period)

    def s0(self, dt):
        """Returns the smallest scale at which wavelet resolution is good"""
        return find_s0(self, dt)

    def optimal_scales(self, dt, dj, n):
        """Returns the wavelet scales at which the time and frequency resolutions are good
        """
        s0 = find_s0(self, dt)
        return find_optimal_scales(s0, dt, dj, n)

########################################################################################################
# Complex Morlet Wavelet
########################################################################################################

def morlet(w0=6, complete=False):
    """
    Returns the n-point continuous Morlet wavelet

    See the definition at https://en.wikipedia.org/wiki/Morlet_wavelet

    w is the center frequency parameter
    a is the scale parameter
    """
    def time(t, s=1.):
        s = jnp.atleast_2d(jnp.asarray(s)).T
        t = t / s
        # wavelet 1 / (pi)^{1/4} e^{j w t / a} e^{-t^2/ a^2}
        output = jnp.exp(1j * w0 * t)
        if complete:
            output = output - jnp.exp(-0.5 * (w0 ** 2))
        output = output * jnp.exp(-0.5 * t**2) * jnp.pi**(-0.25)
        # energy conservation
        output = jnp.sqrt(1/s) * output
        return jnp.squeeze(output)

    def frequency(w, s=1.0):
        s = jnp.atleast_2d(jnp.asarray(s)).T
        x = w * s
        # Heaviside mock
        Hw = (w > 0).astype(float)
        points = (jnp.pi ** -.25) * Hw * jnp.exp((-(x - w0) ** 2) / 2)
        # normalize for scale
        points = (s ** 0.5) * ((2*jnp.pi) ** 0.5) * points
        return jnp.squeeze(points)

    def fourier_period(s):
        s = jnp.asarray(s)
        return 4 * jnp.pi * s / (w0 + (2 + w0 ** 2) ** .5)

    def scale_from_period(period):
        coeff = jnp.sqrt(w0 * w0 + 2)
        return (period * (coeff + w0)) / (4. * jnp.pi)

    def coi(s):
        return 2 ** .5 * s

    return WaveletFunctions(is_complex=True, 
        time=time, frequency=frequency, fourier_period=fourier_period,
        scale_from_period=scale_from_period, coi=coi)




def cmor(B, C):
    """
    Returns the n-point continuous Morlet wavelet

    Args:
        B the bandwidth parameter
        C the central frequency
    """
    def time(t, s=1.):
        s = jnp.atleast_2d(jnp.asarray(s)).T
        t = t / s
        # the sinusoid
        output = jnp.exp(1j * 2 * jnp.pi * C * t)
        # the Gaussian
        output = output * jnp.exp(-t**2 /B ) 
        # the normalization factor
        factor = (jnp.pi *B) **(-0.5)
        output = factor * output
        # energy conservation
        output = jnp.sqrt(1/s) * output
        return jnp.squeeze(output)

    def frequency(w, s=1.0):
        s = jnp.atleast_2d(jnp.asarray(s)).T
        x = w * s
        # Heaviside mock
        Hw = (w > 0).astype(float)
        # subtract angular frequencies with angular central frequency
        x = x - 2*jnp.pi*C
        # apply the bandwidth factor
        x = x * B / 4
        # apply the exponential
        points =  Hw * jnp.exp(-x)
        # normalize for scale
        points = (s ** 0.5) * ((2*jnp.pi) ** 0.5) * points
        return jnp.squeeze(points)

    def fourier_period(s):
        s = jnp.asarray(s)
        return s / C

    def scale_from_period(period):
        period = jnp.asarray(period)
        return period * C

    def coi(s):
        return 2 ** .5 * s

    return WaveletFunctions(is_complex=True, 
        time=time, frequency=frequency, fourier_period=fourier_period,
        scale_from_period=scale_from_period, coi=coi)


########################################################################################################
# Ricker Wavelet
########################################################################################################

def ricker():
    """
    Returns the n-point continuous Ricker/Mexican Hat wavelet function

    See the definition at https://en.wikipedia.org/wiki/Ricker_wavelet
    """
    def time(t, s=1.):
        s = jnp.atleast_2d(jnp.asarray(s)).T
        # The normalization term 2 / (sqrt(3 s) pi^{1/4})
        A = 2 / (jnp.sqrt(3 * s) * (jnp.pi**0.25))
        # square the scale s^2
        wsq = s**2
        # t^2
        xsq = t**2
        # the modulation term (1 - t^2/a^2)
        mod = (1 - xsq / wsq)
        # the gaussian term e^{-t^2/2a^2}
        gauss = jnp.exp(-xsq / (2 * wsq))
        total = A * mod * gauss
        return jnp.squeeze(total)

    def frequency(w, s=1.0):
        s = jnp.atleast_2d(jnp.asarray(s)).T
        x = w * s
        function = x ** 2 * jnp.exp(-x ** 2 / 2)
        # The normalization term 2 / (sqrt(3 s) pi^{1/4})
        A = 2 / (jnp.sqrt(3) * (jnp.pi**0.25))
        result = A * function
        # normalize for scale
        result = (s ** 0.5) * ((2*jnp.pi) ** 0.5) * result
        return jnp.squeeze(result)

    def fourier_period(s):
        s = jnp.asarray(s)
        return 2 * jnp.pi * s / (2.5) ** .5

    def scale_from_period(period):
        raise NotImplementedError()

    def coi(s):
        return 2 ** .5 * s

    return WaveletFunctions(is_complex=False, 
        time=time, frequency=frequency, fourier_period=fourier_period,
        scale_from_period=scale_from_period, coi=coi)
