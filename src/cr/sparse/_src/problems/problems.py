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
