import pytest
import math
# jax imports
import jax
import jax.numpy as jnp

# crs imports
import cr.sparse as crs
from cr.sparse.dsp import signals


def test_chirp():
    fs = 100
    f0 = 1
    f1 = 5
    T = 5
    initial_phase = 0
    t, sig = signals.chirp(fs, T, f0, f1, initial_phase)
    assert len(t) == len(sig)

def test_chirp_centered():
    fs = 100
    f0 = 1
    f1 = 5
    T = 5
    fc = (f0 + f1) / 2
    bw = f1 - f0
    initial_phase = 0
    t, sig = signals.chirp_centered(fs, T, fc, bw, initial_phase)
    assert len(t) == len(sig)

def test_pulse():
    fs = 100
    T = 16
    begin = 4
    end = 6
    init = -4
    t, sig = signals.pulse(fs, T, begin, end, init)
    assert len(t) == len(sig)


def test_gaussian_pulse():
    fs = 1000
    T = 4
    b = T/2
    fc = 5
    t, sig = signals.gaussian_pulse(fs, T, b, fc)
    assert len(t) == len(sig)
    t, real, imag = signals.gaussian_pulse(fs, T, b, fc, retquad=True)
    assert len(t) == len(real)
    t, sig, env = signals.gaussian_pulse(fs, T, b, fc, retenv=True)
    assert len(t) == len(sig)
    t, real, imag, env = signals.gaussian_pulse(fs, T, b, fc, retenv=True, retquad=True)
    assert len(t) == len(real)


def test_decaying_sine_wave():
    fs = 100
    T = 10
    f = 2
    alpha = 0.5
    t, sig = signals.decaying_sine_wave(fs, T, f, alpha)
    assert len(t) == len(sig)


def test_transient_sine_wave():
    fs = 100
    T = 10
    f = 2
    T = 16
    begin = 2
    end = 6
    init = -4
    t, sig = signals.transient_sine_wave(fs, T, f, begin, end, initial_time=init)
    assert len(t) == len(sig)


def test_gaussian():
    fs = 1000
    T = 20
    a = 1
    b = T/2
    t, sig = signals.gaussian(fs, T, b, a=a)
    assert len(t) == len(sig)
