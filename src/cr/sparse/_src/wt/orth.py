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

"""
Orthogonal wavelets
"""
import math

from jax import jit
import jax.numpy as jnp

from .dyad import *
from .transform import *


def wavelet_function(qmf, j, k, n):
    """
    Returns the (j,k)-th wavelet function for an orthogonal wavelet

    Inputs: 
      j, k - scale and location indices
      n - signal length (dyadic)
    """
    w = jnp.zeros(n)
    # identify the index of the j-th scale at k-th translation
    index = 2**j + k
    w = w.at[index].set(1)         
    return inverse_periodized_orthogonal(qmf, w, j)

def scaling_function(qmf, j, k, n):
    """
    Returns the (j,k)-th scaling function for an orthogonal wavelet

    Inputs: 
      j, k - scale and location indices
      n - signal length (dyadic)
    """
    w = jnp.zeros(n)
    # k-th translate in the coarsest part of the wavelet coefficients
    w = w.at[k].set(1)
    return inverse_periodized_orthogonal(qmf, w, j)

def haar():
    """
    Returns the quadrature mirror filter for the Haar wavelets
    """
    return jnp.array([1., 1]) / math.sqrt(2)



def db4():
    return jnp.array([.482962913145, .836516303738, .224143868042, -.129409522551  ])

def db6():
    return jnp.array([.332670552950, .806891509311, .459877502118, 
        -.135011020010, -.085441273882,  .035226291882])

def db8():
    return jnp.array([.230377813309, .714846570553, .630880767930, -.027983769417, 
        -.187034811719, .030841381836, .032883011667, -.010597401785])

def db10():
    return jnp.array([.160102397974,   .603829269797,   .724308528438, 
                .138428145901,   -.242294887066,  -.032244869585,
                .077571493840,   -.006241490213,  -.012580751999,
                .003335725285])

def db12():
    return jnp.array([.111540743350,   .494623890398,   .751133908021, 
                .315250351709,   -.226264693965,  -.129766867567,
                .097501605587,   .027522865530,   -.031582039317,
                .000553842201,   .004777257511,   -.001077301085,])

def db14():
    return jnp.array([.077852054085,   .396539319482,   .729132090846, 
                .469782287405,   -.143906003929,  -.224036184994,
                .071309219267,   .080612609151,   -.038029936935,
                -.016574541631,  .012550998556,   .000429577973,
                -.001801640704,  .000353713800, ])

def db16():
    return jnp.array([.054415842243,   .312871590914,   .675630736297, 
                .585354683654,   -.015829105256,  -.284015542962,
                .000472484574,   .128747426620,   -.017369301002,
                -.044088253931,  .013981027917,   .008746094047,
                -.004870352993,  -.000391740373,  .000675449406, 
                -.000117476784, ])

def db18():
    return jnp.array([.038077947364,   .243834674613,   .604823123690, 
                .657288078051,   .133197385825,   -.293273783279,
                -.096840783223,  .148540749338,   .030725681479, 
                -.067632829061,  .000250947115,   .022361662124, 
                -.004723204758,  -.004281503682,  .001847646883, 
                .000230385764,   -.000251963189,  .000039347320,])

def db20():
    return jnp.array([.026670057901,   .188176800078,   .527201188932, 
                .688459039454,   .281172343661,   -.249846424327,
                -.195946274377,  .127369340336,   .093057364604, 
                -.071394147166,  -.029457536822,  .033212674059, 
                .003606553567,   -.010733175483,  .001395351747, 
                .001992405295,   -.000685856695,  -.000116466855,
                .000093588670,   -.000013264203,])


def baylkin():
    return jnp.array([.099305765374,   .424215360813,   .699825214057, 
            .449718251149,   -.110927598348,  -.264497231446,
            .026900308804,   .155538731877,   -.017520746267,
            -.088543630623,  .019679866044,   .042916387274,
            -.017460408696,  -.014365807969,  .010040411845, 
            .001484234782,   -.002736031626,  .000640485329,])

