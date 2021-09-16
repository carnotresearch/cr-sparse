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
Convolutions 1D, 2D, ND
"""
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp


from .impl import _hermitian
from .lop import Operator
from .util import apply_along_axis


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
    h_conj = _hermitian(h[::-1])
    f_slice = slice(forward, forward+n, None)
    b_slice = slice(adjoint, adjoint+n, None)
    times1d = lambda x : jnp.convolve(x, h, 'full')[f_slice]
    trans1d = lambda x : jnp.convolve(x, h_conj, 'full')[b_slice]
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
    times = lambda x : jsp.signal.convolve(jnp.reshape(x, shape), h, 'full')[f_slices]
    trans = lambda x : jsp.signal.correlate(jnp.reshape(x, shape), h, 'full')[a_slices]
    return Operator(times=times, trans=trans, shape=(shape,shape))
