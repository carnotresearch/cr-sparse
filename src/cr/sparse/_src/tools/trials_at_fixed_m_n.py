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

import time
from typing import NamedTuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

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
    min_iters: int = 0
    max_iters: int = 0
    mean_iters: int = 0
    runtime: float = 0.0
    mean_runtime: float = 0.0

class RecoveryTrialsAtFixed_M_N:
    """Experiment of sparse recovery trials for multiple solvers at fixed dictionary size

    Procedure

    * Setup the experiment parameters
    * Add the solvers to be evaluated under the experiment
    * Run the experiment (specify the CSV file in which the results will be saved)

    The results are saved automatically during the experiment.

    Sample usage::

        from functools import partial
        import jax.numpy as jnp
        from cr.sparse.pursuit.eval import RecoveryTrialsAtFixed_M_N
        from cr.sparse.pursuit import htp

        Ks = jnp.array(list(range(1, 4)) + list(range(4, 60, 8)))
        evaluation = RecoveryTrialsAtFixed_M_N(
            M = 200,
            N = 1000,
            Ks = Ks,
            num_dict_trials = 1,
            num_signal_trials = 20
        )
        # Add multiple solvers
        htp_solve_jit = partial(htp.solve_jit, normalized=False)
        nhtp_solve_jit = partial(htp.solve_jit, normalized=True)
        evaluation.add_solver('HTP', htp_solve_jit)
        evaluation.add_solver('NHTP', nhtp_solve_jit)
        # Run evaluation
        evaluation('record_combined_performance.csv')
    """

    def __init__(self, M, N, Ks, num_dict_trials, num_signal_trials):
        """Initializes the experiment parameters.

        Parameters:

            M: (fixed) signal/measurement space dimension
            N: (fixed) number of atoms / representation space dimension
            Ks: Different values of sparsity levels for which experiments will be run
            num_dict_trials: Number of dictionaries sampled for each value of K
            num_signal_trials: Number of sparse vectors sampled for each sampled dictionary for each K

        """
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

    def __call__(self, destination):
        """
        Runs the smulation
        """
        self.destination = destination
        for solver in self.solvers:
            self._process(solver['name'], solver['solver'])

    def save(self):
        """Saves the experiment results in the CSV file"""
        self.df.to_csv(self.destination, index=False)

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
            start_time = time.perf_counter()
            # Keys for tests
            key, subkey = jax.random.split(key)
            dict_keys = jax.random.split(key, num_dict_trials)
            trials = 0
            successes = 0
            success_rate = 0
            iters = []
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
                    if trials == 0:
                        # first trial is for JIT compilation.
                        # We start time measurement after that.
                        start_time = time.perf_counter()
                    trials += 1
                    success = bool(rp.success)
                    successes +=  rp.success
                    iters.append(sol.iterations)
                    print('+' if success else '-', end='', flush=True)
                print('')
            end_time = time.perf_counter()
            # number of failures
            failures = trials - successes
            # success rate
            success_rate = successes / trials
            iters = np.array(iters)
            runtime=end_time-start_time # in seconds
            mean_runtime=runtime * 1000 / (trials - 1) # in milli seconds
            # summarized information
            row = Row(m=M, n=N, k=K, method=name, 
                trials=trials, successes=successes, 
                failures=failures, success_rate=success_rate,
                min_iters=iters.min(), max_iters=iters.max(), mean_iters=iters.mean(),
                runtime=end_time-start_time, mean_runtime=mean_runtime)
            print(row)
            df.loc[len(df)] = row
            self.save()
        self.save()
   
