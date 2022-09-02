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

"""Total variation linear operator
"""
import jax.numpy as jnp
import cr.nimble as cnb

from .lop import Operator
from .util import apply_along_axis


REGULAR = 'regular'
DIRICHLET = 'dirichlet'
CIRCULAR = 'circular'


def diff_fwd_1d_regular(x):
    append = jnp.array([x[-1]])
    return jnp.diff(x, append=append)

def diff_adj_1d_regular(x):
    x = x.at[-1].set(0)
    prepend = jnp.array([0])
    return jnp.diff(-x, prepend=prepend)

def diff_fwd_1d_dirichlet(x):
    append = jnp.array([0])
    return jnp.diff(x, append=append)

def diff_adj_1d_dirichlet(x):
    x1 = cnb.vec_shift_right(x)
    return x1 - x

def diff_fwd_1d_circular(x):
    append = jnp.array([x[0]])
    return jnp.diff(x, append=append)

def diff_adj_1d_circular(x):
    x1 = cnb.vec_rotate_right(x)
    return x1 - x

def tv(n, kind='regular', axis=0):
    r"""Returns a total variation linear operator for 1D signals

    Args:
        n (int): Dimension of the model space
        kind (str): Boundary condition for handling differences 
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        (Operator): An linear operator which computes the variation in 1D signals

    Note:

        To compute the total variation, we first apply the linear operator 
        and then compute the norm of the variation.
    """
    if kind == REGULAR:
        times = diff_fwd_1d_regular
        trans = diff_adj_1d_regular
    elif kind == DIRICHLET:
        times = diff_fwd_1d_dirichlet
        trans = diff_adj_1d_dirichlet
    elif kind == CIRCULAR:
        times = diff_fwd_1d_circular
        trans = diff_adj_1d_circular
    else:
        raise NotImplementedError(f"The kind {kind} is not supported")
    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(n,n))

def tv2D(shape, kind='regular'):
    r"""Returns a total variation linear operator for 2D images

    Args:
        shape (int): Shape of the input images (model space)
        kind (str): Boundary condition for handling differences 

    Returns:
        (Operator): An linear operator which computes the variation in 2D images

    Note:

        The output is a complex image. The horizontal differences are stored
        in the real part and the vertical differences are stored in the imaginary
        part. 

        To compute the total variation, we first apply the linear operator 
        and then compute the norm of the variation image.
    """
    if kind == REGULAR:
        times1d = diff_fwd_1d_regular
        trans1d = diff_adj_1d_regular
    elif kind == DIRICHLET:
        times1d = diff_fwd_1d_dirichlet
        trans1d = diff_adj_1d_dirichlet
    elif kind == CIRCULAR:
        times1d = diff_fwd_1d_circular
        trans1d = diff_adj_1d_circular
    else:
        raise NotImplementedError(f"The kind {kind} is not supported")

    def times(X):
        """Forward total variation
        """
        # horizontal variation
        Dh = jnp.apply_along_axis(times1d, 1, X)
        # vertical variation
        Dv = jnp.apply_along_axis(times1d, 0, X)
        # combine them to complex output
        return Dh + Dv * 1j

    def trans(X):
        """Adjoint total variation
        """
        # horizontal variation
        Dh = jnp.apply_along_axis(trans1d, 1, X)
        # vertical variation
        Dv = jnp.apply_along_axis(trans1d, 0, X)
        # combine them to complex output
        return Dh + Dv * 1j

    return Operator(times=times, trans=trans, shape=(shape,shape))
