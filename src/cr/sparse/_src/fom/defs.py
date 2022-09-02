# Copyright 2021 CR-Suite Development Team
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


from typing import NamedTuple


import jax.numpy as jnp

class FomOptions(NamedTuple):
    """Options for FOCS driver routine
    """
    nonneg : bool = False
    "Whether output is expected to be non-negative"
    solver : str = 'at'
    "Default first order conic solver"
    max_iters: int = 1000
    "Maximum number of iterations for the solver"
    tol: float = 1e-8
    "Tolerance for convergence"
    L0 : float = 1.
    "Initial estimate of Lipschitz constant"
    Lexact: float = jnp.inf
    "Known bound of Lipschitz constant"
    alpha: float = 0.9
    "Line search increase parameter, in (0,1)"
    beta: float = 0.5
    "Backtracking parameter, in (0,1). No line search if >= 1"
    mu: float = 0
    "Strong convexity parameter"
    maximize : bool = False
    "By default, we attempt minimization of the objective, otherwise maximize"
    saddle: bool = False
    "Indicates if it's a saddle point problem setup by SCD subroutine"

    def __str__(self):
        s = []
        s.append(f'solver={self.solver}')
        s.append(f'max_iters={self.max_iters}')
        s.append(f'tol={self.tol}')
        s.append(f'L0={self.L0}')
        s.append(f'Lexact={self.Lexact}')
        s.append(f'alpha={self.alpha}')
        s.append(f'beta={self.beta}')
        s.append(f'mu={self.mu}')
        return '\n'.join(s)


class FomState(NamedTuple):
    """
    State of the FOCS method
    """
    L : float
    "Lipschitz constant estimate"
    theta: float

    ""
    x: jnp.ndarray
    ""
    A_x : jnp.ndarray
    "A @ x "
    g_Ax : jnp.ndarray
    "gradient of f at A @ x + b"
    g_x : jnp.ndarray
    "A^H (g_Ax)"
    f_x : float
    " f(A_x + b)"
    C_x : float
    "value of nonsmooth function h at x"

    y: jnp.ndarray
    A_y : jnp.ndarray
    g_Ay : jnp.ndarray
    g_y : jnp.ndarray
    f_y : float
    C_y : float

    z : jnp.ndarray
    A_z: jnp.ndarray
    g_Az : jnp.ndarray
    g_z : jnp.ndarray
    f_z : float
    C_z : float
    
    # quantities for convergence check
    norm_x : float
    norm_dx : float
    # counters
    iterations: int

    def __str__(self):
        s = []
        s.append(f'L={self.L:.2f}, theta={self.theta:.2f}')
        s.append(f'f_x={self.f_x:.2f}, f_y={self.f_y:.2f}, f_z={self.f_z:.2f}')
        s.append(f'C_x={self.C_x:.2f}, C_y={self.C_y:.2f}, C_z={self.C_z:.2f}')
        s.append(f'norm_x={self.norm_x:.2f}, norm_dx={self.norm_dx:.2e}')
        s.append(f'iterations={self.iterations}')
        s.append(f'')
        return '\n'.join(s)

    @property
    def at_str(self):
        s = []
        s.append(f'iterations={self.iterations}, L={self.L:.2f}, theta={self.theta:.2f}')
        s.append(f'f_y={self.f_y:.2e}, C_z={self.C_z:.2e}, norm_x={self.norm_x:.2e}, norm_dx={self.norm_dx:.2e}')
        return '\n'.join(s)
