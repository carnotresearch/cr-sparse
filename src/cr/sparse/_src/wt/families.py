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

from enum import Enum

class FAMILY(Enum):
    """An enumeration describing the wavelet families supported in this library
    """
    HAAR = 0
    """Haar wavelets"""

    RBIO = 1
    """Reverse Biorthogonal Wavelets"""
    DB = 2
    """Daubechies Wavelets"""

    SYM = 3
    """Symlets"""
    COIF = 4
    """Coiflets"""
    BIOR = 5
    """Biorthogonal Wavelets"""
    DMEY = 6
    """"Discrete Meyer (FIR Approximation) Wavelets"""
    
    GAUS = 7
    "Gaussian Wavelets"
    MEXH = 8
    "Mexican Hat/Ricker Wavelets"
    MORL = 9
    "Morlet Wavelets"
    CGAU = 10
    "Complex Gaussian Wavelets"
    SHAN = 11
    "Shannon Wavelets"
    FBSP = 12
    "Frequency B-Spline Wavelets"
    CMOR = 13
    "Complex Morlet Wavelets"

def is_discrete_wavelet(name: FAMILY):
    """Returns if the wavelet family is a family of discrete wavelets
    """
    return name.value in [FAMILY.HAAR.value,
    FAMILY.RBIO.value, 
    FAMILY.DB.value, 
    FAMILY.SYM.value, 
    FAMILY.COIF.value, 
    FAMILY.BIOR.value, 
    FAMILY.DMEY.value
    ]


__wname_to_family_order = {
    "haar": (FAMILY.HAAR, 0),

    "db1": (FAMILY.DB, 1),
    "db2": (FAMILY.DB, 2),
    "db3": (FAMILY.DB, 3),
    "db4": (FAMILY.DB, 4),
    "db5": (FAMILY.DB, 5),
    "db6": (FAMILY.DB, 6),
    "db7": (FAMILY.DB, 7),
    "db8": (FAMILY.DB, 8),
    "db9": (FAMILY.DB, 9),

    "db10": (FAMILY.DB, 10),
    "db11": (FAMILY.DB, 11),
    "db12": (FAMILY.DB, 12),
    "db13": (FAMILY.DB, 13),
    "db14": (FAMILY.DB, 14),
    "db15": (FAMILY.DB, 15),
    "db16": (FAMILY.DB, 16),
    "db17": (FAMILY.DB, 17),
    "db18": (FAMILY.DB, 18),
    "db19": (FAMILY.DB, 19),

    "db20": (FAMILY.DB, 20),
    "db21": (FAMILY.DB, 21),
    "db22": (FAMILY.DB, 22),
    "db23": (FAMILY.DB, 23),
    "db24": (FAMILY.DB, 24),
    "db25": (FAMILY.DB, 25),
    "db26": (FAMILY.DB, 26),
    "db27": (FAMILY.DB, 27),
    "db28": (FAMILY.DB, 28),
    "db29": (FAMILY.DB, 29),

    "db30": (FAMILY.DB, 30),
    "db31": (FAMILY.DB, 31),
    "db32": (FAMILY.DB, 32),
    "db33": (FAMILY.DB, 33),
    "db34": (FAMILY.DB, 34),
    "db35": (FAMILY.DB, 35),
    "db36": (FAMILY.DB, 36),
    "db37": (FAMILY.DB, 37),
    "db38": (FAMILY.DB, 38),

    "sym2": (FAMILY.SYM, 2),
    "sym3": (FAMILY.SYM, 3),
    "sym4": (FAMILY.SYM, 4),
    "sym5": (FAMILY.SYM, 5),
    "sym6": (FAMILY.SYM, 6),
    "sym7": (FAMILY.SYM, 7),
    "sym8": (FAMILY.SYM, 8),
    "sym9": (FAMILY.SYM, 9),

    "sym10": (FAMILY.SYM, 10),
    "sym11": (FAMILY.SYM, 11),
    "sym12": (FAMILY.SYM, 12),
    "sym13": (FAMILY.SYM, 13),
    "sym14": (FAMILY.SYM, 14),
    "sym15": (FAMILY.SYM, 15),
    "sym16": (FAMILY.SYM, 16),
    "sym17": (FAMILY.SYM, 17),
    "sym18": (FAMILY.SYM, 18),
    "sym19": (FAMILY.SYM, 19),
    "sym20": (FAMILY.SYM, 20),

    "coif1": (FAMILY.COIF, 1),
    "coif2": (FAMILY.COIF, 2),
    "coif3": (FAMILY.COIF, 3),
    "coif4": (FAMILY.COIF, 4),
    "coif5": (FAMILY.COIF, 5),
    "coif6": (FAMILY.COIF, 6),
    "coif7": (FAMILY.COIF, 7),
    "coif8": (FAMILY.COIF, 8),
    "coif9": (FAMILY.COIF, 9),

    "coif10": (FAMILY.COIF, 10),
    "coif11": (FAMILY.COIF, 11),
    "coif12": (FAMILY.COIF, 12),
    "coif13": (FAMILY.COIF, 13),
    "coif14": (FAMILY.COIF, 14),
    "coif15": (FAMILY.COIF, 15),
    "coif16": (FAMILY.COIF, 16),
    "coif17": (FAMILY.COIF, 17),


    "bior1.1": (FAMILY.BIOR, 11),
    "bior1.3": (FAMILY.BIOR, 13),
    "bior1.5": (FAMILY.BIOR, 15),
    "bior2.2": (FAMILY.BIOR, 22),
    "bior2.4": (FAMILY.BIOR, 24),
    "bior2.6": (FAMILY.BIOR, 26),
    "bior2.8": (FAMILY.BIOR, 28),
    "bior3.1": (FAMILY.BIOR, 31),
    "bior3.3": (FAMILY.BIOR, 33),
    "bior3.5": (FAMILY.BIOR, 35),
    "bior3.7": (FAMILY.BIOR, 37),
    "bior3.9": (FAMILY.BIOR, 39),
    "bior4.4": (FAMILY.BIOR, 44),
    "bior5.5": (FAMILY.BIOR, 55),
    "bior6.8": (FAMILY.BIOR, 68),

    "rbio1.1": (FAMILY.RBIO, 11),
    "rbio1.3": (FAMILY.RBIO, 13),
    "rbio1.5": (FAMILY.RBIO, 15),
    "rbio2.2": (FAMILY.RBIO, 22),
    "rbio2.4": (FAMILY.RBIO, 24),
    "rbio2.6": (FAMILY.RBIO, 26),
    "rbio2.8": (FAMILY.RBIO, 28),
    "rbio3.1": (FAMILY.RBIO, 31),
    "rbio3.3": (FAMILY.RBIO, 33),
    "rbio3.5": (FAMILY.RBIO, 35),
    "rbio3.7": (FAMILY.RBIO, 37),
    "rbio3.9": (FAMILY.RBIO, 39),
    "rbio4.4": (FAMILY.RBIO, 44),
    "rbio5.5": (FAMILY.RBIO, 55),
    "rbio6.8": (FAMILY.RBIO, 68),

    "dmey": (FAMILY.DMEY, 0),

    # "gaus1": (FAMILY.GAUS, 1),
    # "gaus2": (FAMILY.GAUS, 2),
    # "gaus3": (FAMILY.GAUS, 3),
    # "gaus4": (FAMILY.GAUS, 4),
    # "gaus5": (FAMILY.GAUS, 5),
    # "gaus6": (FAMILY.GAUS, 6),
    # "gaus7": (FAMILY.GAUS, 7),
    # "gaus8": (FAMILY.GAUS, 8),

    # "mexh": (FAMILY.MEXH, 0),

    # "morl": (FAMILY.MORL, 0),

    # "cgau1": (FAMILY.CGAU, 1),
    # "cgau2": (FAMILY.CGAU, 2),
    # "cgau3": (FAMILY.CGAU, 3),
    # "cgau4": (FAMILY.CGAU, 4),
    # "cgau5": (FAMILY.CGAU, 5),
    # "cgau6": (FAMILY.CGAU, 6),
    # "cgau7": (FAMILY.CGAU, 7),
    # "cgau8": (FAMILY.CGAU, 8),

    # "shan": (FAMILY.SHAN, 0),

    # "fbsp": (FAMILY.FBSP, 0),

    # "cmor": (FAMILY.CMOR, 0),
}

_family_short_names = [
    "haar", "db", "sym", "coif", "bior", "rbio", "dmey", 
    #"gaus", "mexh", "morl", "cgau", "shan", "fbsp", "cmor"
]

_family_long_names = [
    "Haar", "Daubechies", "Symlets", "Coiflets", "Biorthogonal",
    "Reverse biorthogonal", "Discrete Meyer (FIR Approximation)", 
    # "Gaussian",
    # "Mexican hat wavelet", "Morlet wavelet", "Complex Gaussian wavelets",
    # "Shannon wavelets", "Frequency B-Spline wavelets",
    # "Complex Morlet wavelets"
]

def wname_to_family_order(name):
    """Returns the wavelet family and order from the name
    """
    try:
        if len(name) > 4 and name[:4] in ['cmor', 'shan', 'fbsp']:
            name = name[:4]
        code = __wname_to_family_order[name]
        return code
    except KeyError:
        raise ValueError(f"Unknown wavelet name '{name}', check wavelist() for the "
                         "list of available builtin wavelets.")


def _check_kind(name, kind):
    if kind == 'all':
        return True
    family, order = wname_to_family_order(name)
    is_discrete = is_discrete_wavelet(family)
    if kind == 'discrete':
        return is_discrete
    else:
        return not is_discrete

def families(short=True):
    if short:
        return _family_short_names
    return _family_long_names

def wavelist(family=None, kind='all'):
    """Returns the list of wavelts supported by this library
    """
    if kind not in ('all', 'continuous', 'discrete'):
        raise ValueError(f"Unrecognized value for `kind`: {kind}")

    sorting_list = []  # for natural sorting order
    if family is None:
        for name in __wname_to_family_order:
            if _check_kind(name, kind):
                sorting_list.append((name[:2], len(name), name))
    elif family in _family_short_names:
        for name in __wname_to_family_order:
            if name.startswith(family):
                sorting_list.append((name[:2], len(name), name))
    else:
        raise ValueError(f"Invalid short family name '{family}'.")

    sorting_list.sort()
    wavelets = []
    for x, x, name in sorting_list:
        wavelets.append(name)
    return wavelets
