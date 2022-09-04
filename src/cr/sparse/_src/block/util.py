# Copyright 2022 CR-Suite Development Team
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

def start_locations(n, b):
    """
    """
    # number of blocks
    c = n // b
    # start locations
    return jnp.arange(c) * b

def end_locations(n, b):
    starts = start_locations(n, b)
    return starts + b - 1


def block_lens_from_start_locs(blk_starts, n):
    # identify the length of each block
    blk_lens = jnp.diff(blk_starts)
    blk_lens = jnp.append(blk_lens, n - blk_starts[-1])
    return blk_lens
    
