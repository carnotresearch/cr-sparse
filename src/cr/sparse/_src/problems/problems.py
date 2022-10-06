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

import importlib
import matplotlib.pyplot as plt

from typing import Callable,Tuple, List

PROBLEM_MAP = {
    'heavi-sine:fourier:heavi-side': 'prob001',
    'blocks:haar': 'prob002',
    'cosine-spikes:dirac-dct' : 'prob003',
    'complex:sinusoid-spikes:dirac-fourier' : 'prob004',
    'cosine-spikes:dirac-dct:gaussian' : 'prob005',
    'piecewise-cubic-poly:daubechies:gaussian': 'prob006',
    'signed-spikes:dirac:gaussian': 'prob007',
    'complex:signed-spikes:dirac:gaussian': 'prob008',
    'blocks:heavi-side': 'prob009',
    'blocks:normalized-heavi-side': 'prob010',
    'gaussian-spikes:dirac:gaussian': 'prob011',
    'src-sep-1': 'prob401'
}

import cr.nimble as crn
import jax.numpy as jnp
from cr.nimble.dsp import largest_indices

from .spec import Problem

def names() -> List[str]:
    """Returns the names of available problems
    """
    return sorted(PROBLEM_MAP.keys())

def generate(name:str, key=crn.KEY0, **args) -> Problem:
    """Generates a test problem by its name and problem specific arguments

    Args:
        name (str): The name of the problem to be instantiated.
        key: A PRNG key to be used if the problem requires randomized data
        args (dict): The list of keyword parameters to be passed to the
          problem generation code 

    Returns:
        Problem: An instance of a problem
    """
    assert name in PROBLEM_MAP, "Unrecognized problem"
    problem_id = PROBLEM_MAP[name]
    module = importlib.import_module('.' + problem_id,
        'cr.sparse._src.problems')
    problem = module.generate(key, **args)
    return problem


def plot(problem, height=4):
    """Plots the figures associated with a problem

    This is just a utility function to quickly draw
    all the figures in one shot. You may be interested
    in drawing specific individual figures associated
    with a problem.
    """
    nf = len(problem.figures)
    fig, ax = plt.subplots(nf, 1, figsize=(15, height*nf), 
        constrained_layout = True)
    for i in range(nf):
        problem.plot(i, ax[i])
    return fig, ax


def analyze_solution(problem, solution, perc=99.9):
    """Provides a quick analysis of sparse recovery by one of
    the algorithms in CR-Sparse
    """
    A = problem.A
    b0 = problem.b
    x0 = problem.x
    y0 = problem.y
    m, n = A.shape
    x = solution.x
    b = A.times(x)
    print(f'm: {m}, n: {n}')
    print(f'b_norm: original: {crn.arr_l2norm(b0):.3f} reconstruction: {crn.arr_l2norm(b):.3f}'
        + f' SNR: {crn.signal_noise_ratio(b0, b):.2f} dB')
    if x0 is not None:
        print(f'x_norm: original: {crn.arr_l2norm(x0):.3f} reconstruction: {crn.arr_l2norm(x):.3f}'
            + f' SNR: {crn.signal_noise_ratio(x0, x):.2f} dB')
    if y0 is not None and (problem.both or x0 is None):
        y = problem.reconstruct(x)
        print(f'y_norm: original: {crn.arr_l2norm(y0):.3f} reconstruction: {crn.arr_l2norm(y):.3f}'
            + f' SNR: {crn.signal_noise_ratio(y0, y):.2f} dB')
    if x0 is not None:
        k1 = crn.num_largest_coeffs_for_energy_percent(x0, perc)
        s1 = largest_indices(x0, k1)
        k2 = crn.num_largest_coeffs_for_energy_percent(x, perc)
        s2 = largest_indices(x, k2)
        overlap = jnp.intersect1d(s1, s2)
        correct = overlap.size
        ratio = correct / max(k1, k2)
        print(f'Sparsity: original: {k1}, reconstructed: {k2}, overlap: {correct}, ratio: {ratio:.3f}')
    if hasattr(solution, 'iterations'):
        print(f'Iterations: {solution.iterations} ', end='')
    if hasattr(solution, 'n_times'):
        print(f'n_times: {solution.n_times}, n_trans: {solution.n_trans}', end='')
    print('\n')

