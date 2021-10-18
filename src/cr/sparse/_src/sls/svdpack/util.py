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

from jax import lax, jit, vmap, random
import jax.numpy as jnp
from jax.numpy.linalg import norm

def do_elr(v_prev, v, v_norm, gamma):
    """
    Extended local reorthogonalization
    """
    def init():
        t = jnp.vdot(v_prev, v)
        v2 = v - t * v_prev
        v2_norm = norm(v2)
        return v2, v2_norm, v_norm, t

    def body(state):
        v, old_norm, older_norm, proj = state
        t = jnp.vdot(v_prev, v)
        v = v - t * v_prev
        v_norm = norm(v)
        return v, v_norm, old_norm, proj + t

    def cond(state):
        v, v_norm, old_norm, proj = state
        return v_norm < gamma * old_norm

    # state = init()
    # while cond(state):
    #     state = body(state)
    state = lax.while_loop(cond, body, init())
    v, v_norm, old_norm, proj = state
    return v, v_norm, proj

def do_elr_noop(v, v_norm):
    """No-op version of ELR for conditional execution"""
    return v, v_norm, 0.