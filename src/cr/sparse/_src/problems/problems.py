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


PROBLEM_MAP = {
    'heavi-sine:fourier:heavi-side': 'prob001',
    'blocks:haar': 'prob002',
    'cosine-spikes:dirac-dct' : 'prob003',
    'complex:sinusoid-spikes:dirac-fourier' : 'prob004',
    'cosine-spikes:dirac-dct:gaussian' : 'prob005',
    'piecewise-cubic-poly:daubechies:gaussian': 'prob006'
}

import cr.nimble as crn

def generate(name, key=crn.KEY0, **args):
    problem_id = PROBLEM_MAP[name]
    module = importlib.import_module('.' + problem_id,
        'cr.sparse._src.problems')
    problem = module.generate(key, **args)
    return problem
