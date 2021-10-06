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

"""Helper functions for solving dense linear systems
"""

from jax.scipy.linalg import solve
from jax.numpy.linalg import lstsq

def mult_with_submatrix(A, columns, x):
    """Computes :math:`b = A[:, I] x`
    """
    A = A[:, columns]
    return A @ x


def solve_on_submatrix(A, columns, b):
    """Solves the problem :math:`A[:, I] x = b` where I is an
    index set of selected columns
    """
    A = A[:, columns]
    x, r_norms, rank, s = lstsq(A, b)
    r = b - A @ x
    return x, r

