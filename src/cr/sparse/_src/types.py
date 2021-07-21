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
A collection of data types useful for various problems.
"""
import jax.numpy as jnp
from typing import NamedTuple, List, Dict


class RecoveryFullSolution(NamedTuple):
    """Represents the solution of a sparse recovery problem

    Consider a sparse recovery problem :math:`b=A x + e`.

    This type combines all of this information together.

    Parameters:

        x : estimate(s) of :math:`x`
        r : residual(s) :math:`r = b - A x `
        iterations: Number of iterations required for the algorithm to converge

    Note:

        The tuple can be used to solve multiple measurement vector
        problems also. In this case, each column (of individual parameters)
        represents the solution of corresponding single vector problems.
    """
    # The non-zero values
    x: jnp.ndarray
    """Solution vector"""
    r: jnp.ndarray
    """The primal residual vector"""
    iterations: int
    """The number of iterations it took to complete"""
    n_times: int = 0
    """Number of times A x computed """
    n_trans : int = 0
    """Number of times A^H b computed """
