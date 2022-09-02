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

"""Spread operator
"""
import numpy as np
import jax.numpy as jnp
from .lop import Operator
from .util import apply_along_axis


def spread_with_table(in_dims, out_dims, table, dtable):
    """Returns a spread linear operator using a table
    """
    nx_in, nt_in = in_dims
    nx_out, nt_out = out_dims
    if table.shape != (nx_in, nt_in, nx_out):
        raise ValueError(f"Table must have shape ({nx_in}, {nt_in}, {nx_in})")
    if jnp.any(table > nt_out):
        raise ValueError(f"All values in the table must be smaller than {nt_out}")
    if dtable.shape != (nx_in, nt_in, nx_out):
        raise ValueError(f"Delta table must have shape ({nx_in}, {nt_in}, {nx_in})")

    def times(x):
        """Forward operation

        Every source index is mapped to a row in the table.
        - The table row has some nan entries and some regular entries.
        - Think of the row has (index, value) pairs.
        - index identifies output row
        - value identifies output column
        - If value is nan, then the corresponding output row is skipped.
        """
        # Ensure that input has the correct dimensions
        x = jnp.reshape(x, in_dims)
        # prepare the output array with zeros
        y = jnp.zeros(out_dims, dtype=x.dtype)
        # iterate over input dimensions
        for it_in in range(nt_in):
            for ix_in in range(nx_in):
                indices = table[ix_in, it_in]
                dindices = dtable[ix_in, it_in]
                not_nan = ~jnp.isnan(indices)
                indices = (indices).astype(np.int)
                value = x[ix_in, it_in]
                y = y.at[:, indices].add(value * not_nan)
        return y

    def trans(x):
        """Inverse operation
        """
        pass
    return Operator(times=times, trans=trans, shape=(out_dims,in_dims))
