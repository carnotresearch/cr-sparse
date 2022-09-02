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

import jax.numpy as jnp

from .lop import Operator

import cr.nimble as cnb



def dot(v, adjoint=False, axis=0):
    """Returns a linear operator T such that :math:`T x = \\langle v , x \\rangle = v^H x`

    Args:
        v (jax.numpy.ndarray): The vector/array with which the inner product will be computed 
        adjoint (bool): Indicates if we need the dot operator or its adjoint
        axis (int): For multi-dimensional array input, the axis along which
        the linear operator will be applied 

    Note: 
        axis parameter is useful only if v is 1D. 
    """
    v = jnp.asarray(v)
    assert v.ndim >= 1, "v cannot be a scalar"
    # make sure that v is inexact
    v = cnb.promote_arg_dtypes(v)
    n = v.shape[0] if v.ndim == 1 else v.shape 
    m = 1

    def times1d(x):
        result = cnb.arr_rdot(v, x)
        return jnp.expand_dims(result, 0)

    def times(x):
        x = jnp.asarray(x)
        assert x.shape == v.shape, "shape of x must be same as shape of v"
        if x.ndim == v.ndim:
            return times1d(x)
        if v.ndim == 1:
            return jnp.apply_along_axis(times1d, axis, x)
        raise ValueError("axis parameter is not supported for ND arrays")

    def trans(x):
        # the inner product must be real
        assert jnp.isrealobj(x), "The data space is real as this linear operator represents a real inner product"
        return v * x

    if adjoint:
        m,n = n, m
        times, trans = trans, times

    return Operator(times=times, trans=trans, shape=(m, n))
