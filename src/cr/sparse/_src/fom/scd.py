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

"""
First Order Solver for Smooth Conic Dual Problem
"""

from .defs import FomOptions, FomState


import jax.numpy as jnp
from jax import jit, lax

import cr.nimble as cnb
import cr.sparse.opt as opt
import cr.sparse.lop as lop

from .fom import fom

def smooth_dual(prox_f: opt.ProxCapable, mu=1, x0=0):
    """Constructs the smooth dual of a prox capable function
    """

    @jit
    def func(x):
        """Computes the value of the function at x
        """
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        px, pv = prox_f.prox_vec_val(x0 + mu * x, mu)
        v = cnb.arr_rdot(x, px) - pv - (0.5/mu) * cnb.arr_rnorm_sqr(px - x0)
        return -v

    @jit
    def grad(x):
        """Computes the gradient of the smooth function at x"""
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        px = prox_f.prox_op(x0 + mu * x, mu)
        return -px

    @jit
    def grad_val(x):
        """Computes the gradient as well as the value of the function at x"""
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        px, pv = prox_f.prox_vec_val(x0 + mu * x, mu)
        # the gradient
        g = -px
        v = cnb.arr_rdot(x, px) - pv - (0.5/mu) * cnb.arr_rnorm_sqr(px - x0)
        v = -v
        return g,v



    return opt.smooth_build3(func, grad, grad_val)
    


def scd(prox_f: opt.ProxCapable, conj_neg_h: opt.ProxCapable, 
    A: lop.Operator, b, mu, x0, z0, options: FomOptions = FomOptions()):
    r"""First order solver for smooth conic dual problems driver routine

    Args:
        prox_f (cr.sparse.opt.SmoothFunction): A prox-capable objective function 
        conj_neg_h (cr.sparse.opt.ProxCapable): The conjugate negative :math:`h^{-}` function 
        A (cr.sparse.lop.Operator): A linear operator 
        b (jax.numpy.ndarray): The translation vector
        mu (float): The (positive) scaling term for the quadratic term :math:`\frac{\mu}{2} \| x - x_0 \|_2^2` 
        x0 (jax.numpy.ndarray): The center point for the quadratic term
        z0 (jax.numpy.ndarray): The initial dual point 
        options (FomOptions): Options for configuring the algorithm

    Returns: 
        FomState: Solution of the optimization problem

    The function uses first order conic solver algorithms to solve an
    optimization problem of the form:

    .. math::

        \underset{x}{\text{minimize}} 
        \left [ f(x) + \frac{\mu}{2} \| x - x_0 \|_2^2 
        + h \left (\AAA(x) + b \right) \right ]

    * Both :math:`f, h` must be convex and prox-capable, although neither needs to be smooth.

    When :math:`h` is an indicator function for a convex cone :math:`\KKK`, this is 
    equivalent to:

    .. math::

        \begin{split}\begin{aligned}
        & \underset{x}{\text{minimize}}
        & &  f(x) + \frac{\mu}{2} \| x - x_0 \|_2^2\\
        & \text{subject to}
        & &  \AAA(x) + b \in \KKK
        \end{aligned}\end{split}

    which is the smooth conic dual (SCD) model discussed in  :cite:`becker2011templates`.
    """
    smooth_f = smooth_dual(prox_f, mu, x0)
    options  = options._replace(saddle=True, maximize=True)
    sol = fom(smooth_f, conj_neg_h, A, b, z0, options)
    return sol
