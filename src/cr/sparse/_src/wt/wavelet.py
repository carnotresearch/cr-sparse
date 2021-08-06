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


from enum import Enum, auto
from typing import NamedTuple, List, Dict, Tuple

import jax.numpy as jnp

from .coeffs import db, sym, coif, bior, dmey

class SYMMETRY(Enum):
    UNKNOWN = -1
    ASYMMETRIC = 0
    NEAR_SYMMETRIC = 1
    SYMMETRIC = 2
    ANTI_SYMMETRIC = 3


class FAMILY(Enum):
    HAAR = 0
    RBIO = 1
    DB = 2
    SYM = 3
    COIF = 3
    BIOR = 5
    DMEY = 6
    GAUS = 7
    MEXH = 8
    MORL = 9
    CGAU = 10
    SHAN = 11
    FBSP = 12
    CMOR = 13

def is_discrete_wavelet(name: FAMILY):
    return name.value in [FAMILY.HAAR.value,
    FAMILY.RBIO.value, 
    FAMILY.DB.value, 
    FAMILY.SYM.value, 
    FAMILY.COIF.value, 
    FAMILY.BIOR.value, 
    FAMILY.DMEY.value
    ]

class BaseWavelet(NamedTuple):
    """Represents basic information about a wavelet
    """
    support_width: int = 0
    symmetry: SYMMETRY = SYMMETRY.UNKNOWN
    orthogonal: bool = False
    biorthogonal: bool = False
    compact_support: bool = False
    name: FAMILY = None
    family_name: str = None
    short_name: str = None

class DiscreteWavelet(NamedTuple):
    """Represents information about a discrete wavelet
    """
    support_width: int = -1
    """Length of the support for finite support wavelets"""
    symmetry: SYMMETRY = SYMMETRY.UNKNOWN
    """Indicates the kind of symmetry inside the wavelet"""
    orthogonal: bool = False
    """Indicates if the wavelet is orthogonal"""
    biorthogonal: bool = False
    """Indicates if the wavelet is biorthogonal"""
    compact_support: bool = False
    """Indicates if the wavelet has compact support"""
    name: FAMILY = None
    """Name of the wavelet family"""
    family_name: str = None
    """Name of the wavelet family"""
    short_name: str = None
    """Short name of the wavelet family"""
    dec_hi: jnp.DeviceArray = None
    """Decomposition high pass filter"""
    dec_lo: jnp.DeviceArray = None
    """Decomposition low pass filter"""
    rec_hi: jnp.DeviceArray = None
    """Reconstruction high pass filter"""
    rec_lo: jnp.DeviceArray = None
    """Reconstruction low pass filter"""
    dec_len: int = 0
    """Length of decomposition filters"""
    rec_len: int = 0
    """Length of reconstruction filters"""
    vanishing_moments_psi: int = 0
    """Number of vanishing moments of the wavelet function"""
    vanishing_moments_phi: int = 0
    """Number of vanishing moments of the scaling function"""


def mirror(h):
    n = h.shape[0]
    modulation = (-1)**jnp.arange(1, n+1)
    return modulation * h

def build_discrete_wavelet(name: FAMILY, order: int):
    nv = name.value
    if nv is FAMILY.HAAR.value:
        qmf = db[0]
        dec_hi = mirror(qmf)
        dec_lo = qmf[::-1]
        rec_lo = qmf
        rec_hi = dec_hi[::-1]
        w = DiscreteWavelet(support_width=1,
            symmetry=SYMMETRY.ASYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name=name,
            family_name = "Haar",
            short_name="haar", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=2,
            rec_len=2,
            vanishing_moments_psi=1,
            vanishing_moments_phi=0)
        return w
    if nv == FAMILY.DB.value:
        index = order - 1
        if index >= len(db):
            return None
        filters_length = 2 * order
        dec_len = filters_length
        rec_len = filters_length
        qmf = db[index]
        dec_hi = mirror(qmf)
        dec_lo = qmf[::-1]
        rec_lo = qmf
        rec_hi = dec_hi[::-1]
        w = DiscreteWavelet(support_width=2*order-1,
            symmetry=SYMMETRY.ASYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name=name,
            family_name = "Daubechies",
            short_name="db", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=order,
            vanishing_moments_phi=0)
        return w
    if nv == FAMILY.SYM.value:
        index = order - 2
        if index >= len(sym):
            return None
        filters_length = 2 * order
        dec_len = filters_length
        rec_len = filters_length
        qmf = sym[index]
        dec_hi = mirror(qmf)
        dec_lo = qmf[::-1]
        rec_lo = qmf
        rec_hi = dec_hi[::-1]
        w = DiscreteWavelet(support_width=2*order-1,
            symmetry=SYMMETRY.NEAR_SYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name=name,
            family_name = "Symlets",
            short_name="sym", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=order,
            vanishing_moments_phi=0)
        return w
    return None

