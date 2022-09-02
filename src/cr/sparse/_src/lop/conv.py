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
Convolutions 1D, 2D, ND
"""
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax


from .impl import _hermitian
from .lop import Operator
from .util import apply_along_axis
from cr.nimble import promote_arg_dtypes

def convolve(n, h, offset=0, axis=0):
    """Implements a convolution operator with the filter h

    Note:

        We don't use padding of coefficients of h. It turns out,
        it is faster to perform a full convolution and then 

    """
    assert n > 0
    m = len(h)
    # The location of center of the filter response should be within it.
    assert offset >= 0
    assert offset < m
    forward = offset
    adjoint = m  - 1 - offset
    # for the adjoint, we will simply use the conjugate of h
    h_conj = jnp.conjugate(h)
    # invert the entries as lax convolve is actually correlation
    h = h[slice(None, None, -1)]
    f_slice = slice(forward, forward+n, None)
    b_slice = slice(adjoint, adjoint+n, None)
    
    # add additional dimensions to h
    h =  h[None, None, None]
    h_conj = h_conj[None, None, None]
    # padding for conv_general_dilated
    padding = [(0, 0), (m - 1, m - 1)]
    # strides for conv_general_dilated
    strides = (1,1)

    def times1d(x):
        """Forward convolution
        """
        x, f = promote_arg_dtypes(x, h)
        result = lax.conv_general_dilated(x[None, None, None], f, strides, 
            padding)
        return result[0, 0, 0, f_slice]

    def trans1d(x):
        """Adjoint convolution
        """
        x, f = promote_arg_dtypes(x, h_conj)
        result = lax.conv_general_dilated(x[None, None, None], f, strides, 
            padding)
        return result[0, 0, 0, b_slice]

    times, trans = apply_along_axis(times1d, trans1d, axis)
    return Operator(times=times, trans=trans, shape=(n,n))

def convolve2D(shape, h, offset=None, axes=None):
    """Performs 2 dimensional convolution on the input array
    """
    N = h.ndim
    # The filter must be two dimensional
    assert N == 2
    # Implemented in terms of N dimensional convolution
    return convolveND(shape, h, offset, axes)


def convolveND(shape, h, offset=None, axes=None):
    """Performs N dimensional convolution on input array
    """
    # The dimensions of the filter
    filter_ndim = h.ndim
    # The dimensions of the data
    data_ndim = len(shape)
    # By default offset along each filter dimension is 0
    if offset is None:
        offset = np.zeros(filter_ndim, dtype=int)
    else:
        offset = np.array(offset)
    # offset dimensions must match the filter dimensions
    assert offset.size == filter_ndim
    if axes is None:
        # By default, the convolution will happen over the first filter_ndim dimensions
        axes = np.arange(filter_ndim)
    else:
        axes = np.array(axes)
    # prepare the slices to be extracted from convolution results
    f_slices = [slice(None) for _ in range(data_ndim)]
    a_slices = [slice(None) for _ in range(data_ndim)]
    for i, ax in enumerate(axes):
        # offset along i-th axis
        off_ax = offset[i]
        # the filter size for i-th axis
        h_ax = h.shape[i]
        # the data size for i-th axis
        n_ax = shape[ax]
        # the offset for the adjoint operator for i-th axis
        adj_ax = h_ax - 1 - off_ax
        # the forward slice for the i-th axis
        f_slices[ax] = slice(off_ax, off_ax + n_ax)
        # the adjoint slice for the i-th axis
        a_slices[ax] = slice(adj_ax, adj_ax + n_ax)
    # print(f_slices)
    # print(a_slices)
    f_slices = tuple(f_slices)
    a_slices = tuple(a_slices)
    # check if filter dimensions and data dimensions match
    if data_ndim != filter_ndim:
        # Extend the filter to the data dimensions
        h_dims = np.ones(data_ndim, dtype=int)
        h_dims[axes] = h.shape
        h = jnp.reshape(h, h_dims)
    
    padding = [(s - 1, s - 1) for s in h.shape]
    strides = tuple(1 for s in h.shape)
    # reverse the h kernel
    h_conv = h[tuple(slice(None, None, -1) for s in shape)]
    # extend it
    h_conv = h_conv[None, None]
    h_corr = h[None, None]

    def times(x):
        """Forward N-D convolution
        """
        x, f = promote_arg_dtypes(x, h_conv)
        # Make sure that x has the appropriate shape
        x = jnp.reshape(x, shape)
        result = lax.conv_general_dilated(x[None, None], f, strides, 
            padding)
        result = result[0, 0]
        # pick the slices from other dimensions
        return result[f_slices]

    def trans(x):
        """Backward N-D convolution
        """
        x, f = promote_arg_dtypes(x, h_corr)
        # Make sure that x has the appropriate shape
        x = jnp.reshape(x, shape)
        result = lax.conv_general_dilated(x[None, None], f, strides, 
            padding)
        result = result[0, 0]
        # pick the slices from other dimensions
        return result[a_slices]

    return Operator(times=times, trans=trans, shape=(shape,shape))
