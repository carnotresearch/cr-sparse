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

import jax.numpy as jnp

def homogenize_vec(x, value=1):
    assert x.ndim == 1
    return jnp.hstack((x, value))

def homogenize_cols(X, value=1):
    assert X.ndim == 2
    n = X.shape[-1]
    o = value * jnp.ones(n)
    return jnp.vstack((X, o))


def homogenize(X, value=1):
    if X.ndim == 1:
        return homogenize_vec(X, value)
    return homogenize_cols(X, value)