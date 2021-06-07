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

import jax
import jax.numpy as jnp
import pandas as pd
from typing import NamedTuple

import cr.sparse as crs
from cr.sparse import pursuit
import cr.sparse.data as crdata
import cr.sparse.dict as crdict

from .performance import RecoveryPerformance


class Row(NamedTuple):
    method: str
    m : int
    n : int
    k : int
    trials: int = 0
    successes : int = 0
    failures: int = 0
    success_rate: float = 0

class SuccessRates:

    def __init__(self, M, N, Ks, num_dict_trials=10, num_signal_trials=5):
        self.M  = M
        self.N  = N
        self.Ks = Ks
        self.num_dict_trials = num_dict_trials
        self.num_signal_trials = num_signal_trials
        self.solvers = []
        self.df = pd.DataFrame(columns=Row._fields)

    def add_solver(self, name, solver):
        self.solvers.append({
            "name" : name,
            "solver" : solver
        })

    def __call__(self):
        """
        Runs the smulation
        """
        for solver in self.solvers:
            self._process(solver['name'], solver['solver'])

    def save(self, destination='record_success_rates.csv'):
        self.df.to_csv(destination, index=False)

    def _process(self, name, solver):
        # Copy configuration
        M = self.M
        N = self.N
        Ks = self.Ks
        num_dict_trials = self.num_dict_trials
        num_signal_trials = self.num_signal_trials
        df = self.df
        # Seed the JAX random number generator
        key = jax.random.PRNGKey(0)
        for K in Ks:
            print(f'\nK={K}')
            # Keys for tests
            key, subkey = jax.random.split(key)
            dict_keys = jax.random.split(key, num_dict_trials)
            trials = 0
            successes = 0
            success_rate = 0
            # Iterate over number of trials (dictionaries * signals)
            for ndt in range(num_dict_trials):
                dict_key = dict_keys[ndt]
                print('*', end='', flush=True)
                # Construct a dictionary
                Phi = crdict.gaussian_mtx(dict_key, M,N)
                signal_keys = jax.random.split(dict_key, num_dict_trials)
                for nst in range(num_signal_trials):
                    signal_key = signal_keys[nst]
                    # Construct a signal
                    x, omega = crdata.sparse_normal_representations(signal_key, N, K, 1)
                    x = jnp.squeeze(x)
                    # Compute the measurements
                    y = Phi @ x
                    # Run the solver
                    sol = solver(Phi, y, K)
                    # Measure recovery performance
                    rp = RecoveryPerformance(Phi, y, x, sol=sol)
                    trials += 1
                    success = bool(rp.success)
                    successes +=  rp.success
                    print('+' if success else '-', end='', flush=True)
                print('')
            # number of failures
            failures = trials - successes
            # success rate
            success_rate = successes / trials
            # summarized information
            row = Row(m=M, n=N, k=K, method=name, 
                trials=trials, successes=successes, success_rate=success_rate)
            print(row)
            df.loc[len(df)] = row
   
