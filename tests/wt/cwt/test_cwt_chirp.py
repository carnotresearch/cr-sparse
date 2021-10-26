from cwt_setup import *

fs = 100
T = 10
f0 = 1
f1 = 4
t, x = signals.chirp(fs, T, f0, f1, initial_phase=0)
# voices per octave
nu = 8
scales = wt.scales_from_voices_per_octave(nu, jnp.arange(5))
scales = tuple(scales)


def test_voices_per_octave():
    assert len(scales) == 5
    assert scales[0] == 1

@pytest.mark.parametrize("wavelet,method, approach", [
    ('mexh', 'conv', 'pywt'),
    ('cmor1.5-2.0', 'conv', 'pywt'),
    ('mexh', 'conv', 'tc'),
    ('cmor1.5-2.0', 'conv', 'tc'),
    ('mexh', 'fft', 'tc'),
    ('cmor1.5-2.0', 'fft', 'tc'),
    ])
def test_cwt_mexh_id(wavelet,method, approach):
    precision=4
    output = wt.cwt(x, scales, wavelet=wavelet, 
        method=method, approach=approach, precision=precision)

def test_cwt_invalid_approach():
    with assert_raises(NotImplementedError):
        wt.cwt(x, scales, wavelet='mexh', 
        method='conv', approach='abcd')


@pytest.mark.parametrize("frequency", [True, False])
def test_analyze(frequency):
    analysis = wt.analyze(x)
    coi = analysis.coi
    n = analysis.n
    t = analysis.times
    period = analysis.fourier_period
    scale = analysis.scale_from_period
    periods = analysis.fourier_periods
    frequencies = analysis.fourier_frequencies
    s0 = analysis.s0
    w_k = analysis.w_k
    magnitude = analysis.magnitude
    power = analysis.power
    coi = analysis.coi
    wavelet_transform_delta = analysis.wavelet_transform_delta
    C_d = analysis.C_d
