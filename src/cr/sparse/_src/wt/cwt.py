


from .cwt_tc import cwt_tc
from .cwt_int_diff import cwt_id

def cwt(data, scales, wavelet, sampling_period=1., method='conv', axis=-1, precision=10, approach="pywt"):
    """Computes the CWT of data along a specified axis with a specified wavelet
    """
    if approach == 'pywt':
        return cwt_id(data, scales, wavelet, method=method, axis=axis, precision=precision)
    elif approach == 'tc':
        return cwt_tc(data, scales, wavelet, method=method, axis=axis)
    raise NotImplementedError("The specified approach is not supported yet")
