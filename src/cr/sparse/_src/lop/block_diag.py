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

from .util import apply_along_axis

def block_diag(operators, axis=0):
    r"""Returns a block diagonal operator from 2 or more operators

    Args:
        operators (list): List of linear operators 
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A block diagonal operator

    Assume a set of operators :math:`T_1, T_2, \dots, T_k` 
    each having the shape :math:`(m_i, n_i)`.

    The input model space dimension for the block diagonal operator is:

    .. math::

        n = \sum_{i=1}^k n_i

    The output data space dimension for the block diagonal operator is:

    .. math::

        m = \sum_{i=1}^k m_i

    For the forward mode, split the input model vector :math:`x \in \mathbb{F}^n` into 

    .. math::

        x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_k \end{bmatrix}

    such that :math:`\text{dim}(x_i) = n_i`.

    Then the application of the block diagonal operator 
    in the forward mode from model space to data space
    can be represented as: 

    .. math::

       T x =  \begin{bmatrix}
       T_1 & 0 & \dots & 0 \\
       0 & T_2 & \dots & 0 \\
       \vdots & \vdots & \ddots & \vdots \\
       0  & 0 & \dots &  T_k 
       \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_k \end{bmatrix}
       = \begin{bmatrix} T_1 x_1 \\ T_2 x_2 \\ \vdots \\ T_k x_k \end{bmatrix}

    Similarly the application in the adjoint mode from the
    data space to the model space can be represented as:

    .. math::

       T^H y =  \begin{bmatrix}
       T_1^H & 0 & \dots & 0 \\
       0 & T_2^H & \dots & 0 \\
       \vdots & \vdots & \ddots & \vdots \\
       0  & 0 & \dots &  T_k^H 
       \end{bmatrix}\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_k \end{bmatrix}
       = \begin{bmatrix} T_1^H y_1 \\ T_2^H y_2 \\ \vdots \\ T_k^H y_k \end{bmatrix}

    Examples:
        >>> T1 = lop.matrix(2.*jnp.ones((2,2)))
        >>> T2 = lop.matrix(3.*jnp.ones((3,3)))
        >>> T = lop.block_diag([T1, T2])
        >>> print(lop.to_matrix(T))
        [[2. 2. 0. 0. 0.]
        [2. 2. 0. 0. 0.]
        [0. 0. 3. 3. 3.]
        [0. 0. 3. 3. 3.]
        [0. 0. 3. 3. 3.]]
        >>> x = jnp.arange(5)+1
        >>> print(T.times(x))
        [ 6.  6. 36. 36. 36.]
        >>> print(T.trans(x))
        [ 6.  6. 36. 36. 36.]

    
    """
    assert isinstance(operators, list)
    assert len(operators) >= 2
    in_slices = []
    out_slices = []
    m_all = 0
    n_all = 0
    in_start = 0
    out_start = 0
    for op in operators:
        m, n = op.shape
        m_all += m
        n_all += n
        in_slice = slice(in_start, in_start + n)
        in_slices.append(in_slice)
        out_slice = slice(out_start, out_start + m)
        out_slices.append(out_slice)
        in_start += n
        out_start += m
    # number of operators    
    num_operators = len(operators)

    def times(x):
        """Forward operation"""
        ys = []
        for i in range(num_operators):
            op = operators[i]
            in_slice = in_slices[i]
            out = op.times(x[in_slice])
            ys.append(out)
        return jnp.concatenate(ys)

    def trans(x):
        """Adjoint operation"""
        ys = []
        for i in range(num_operators):
            op = operators[i]
            out_slice = out_slices[i]
            out = op.trans(x[out_slice])
            ys.append(out)
        return jnp.concatenate(ys)

    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(m_all,n_all))
