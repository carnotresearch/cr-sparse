import pytest
from cr.sparse.io.resource import *


def test_get_uri():
    assert get_uri('abc') is None
    assert get_uri("haarcascade_frontalface_default.xml") is not None

def test_ensure_resource():
    fname = "sst_nino3.dat"
    path = CACHE_DIR / fname
    if path.is_file():
        path.unlink()
    path = ensure_resource(fname, "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat")
    assert path.is_file()
    path = ensure_resource("http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat")
    assert path.is_file()
    path = ensure_resource(None)
    assert path is None
    path = ensure_resource('abc')
    assert path is None
