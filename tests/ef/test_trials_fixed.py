import pytest

import jax.numpy as jnp

from functools import partial

from cr.sparse.ef import RecoveryTrialsAtFixed_M_N

def test():
    Ks = jnp.array([1,2])

    evaluation = RecoveryTrialsAtFixed_M_N(
        M = 5,
        N = 10,
        Ks = Ks,
        num_dict_trials = 1,
        num_signal_trials = 2
    )
    # Add solvers
    from cr.sparse.pursuit import htp
    htp_solve_jit = partial(htp.matrix_solve_jit, normalized=False)

    evaluation.add_solver('HTP', htp_solve_jit)
    # Run evaluation
    evaluation('record_combined_performance.csv')
