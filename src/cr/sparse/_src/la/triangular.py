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


from jax.scipy.linalg import solve_triangular

def solve_Lx_b(L, b):
    """
    Solves the system L x = b using back substitution
    """
    return solve_triangular(L, b, lower=True)

def solve_LTx_b(L, b):
    """
    Solves the system L^T x = b using back substitution
    """
    return solve_triangular(L, b, lower=True, trans='T')

def solve_Ux_b(U, b):
    """
    Solves the system U x = b using back substitution
    """
    return solve_triangular(U, b)

def solve_UTx_b(U, b):
    """
    Solves the system U^T x = b using back substitution
    """
    return solve_triangular(U, b, trans='T')


def solve_spd_chol(L, b):
    """
    Solves a symmetric positive definite system A x = b
    where A = L L'
    """
    # We have to solve L L' x = b
    # We first solve L u = b
    u = solve_Lx_b(L, b)
    # We now solve L' x = u
    x = solve_LTx_b(L, u)
    return x
