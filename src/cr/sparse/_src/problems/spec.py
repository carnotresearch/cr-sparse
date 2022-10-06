# Copyright 2022 CR-Suite Development Team
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


"""Basic definitions for problems
"""

import math

from typing import NamedTuple, Callable,Tuple, List

from jax import lax, jit

import jax.numpy as jnp


import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.lop as crlop

class Problem(NamedTuple):
    r"""A sparse signal recovery problem

    The problem is given by
    :math:`\bb = \Phi \by`
    where :math:`\by = \Psi \bx`.
    This can written as :math:`\bb = \Phi \Psi \bx`.
    We shorten :math:`\bA = \Phi \Psi` and
    reformulate the problem as `\bb = \bA \bx`.
    In the sparse reconstruction problem, we have
    access to :math:`\bb` and :math:`\bA`
    and we attempt to recover :math:`\bx`.
    We can reconstruct :math:`\by` from :math:`\bx`
    by using the ``reconstruct`` function provided.
    """
    name: str
    "Name of the problem"
    Phi: crlop.Operator
    "A linear operator representing the sensing process"
    Psi: crlop.Operator
    "A sparsifying basis/dictionary"
    A: crlop.Operator
    "The combined sensing matrix + sparsifying dictionary operator"
    b: jnp.ndarray
    "The observed signal"
    reconstruct: Callable
    "Function handle to reconstruct a signal from coefficients in x"
    x: jnp.ndarray = None
    "Expected sparse representation (if available for synthetic problems)"
    y: jnp.ndarray = None
    "Original signal"
    figures : List[str] = []
    "Titles of figures associated with the problem"
    plot: Callable = None
    "A function to plot specific figures associated with the problem"
    both: bool = False
    "A flag to indicate if both Phi and Psi are non-trivial"


