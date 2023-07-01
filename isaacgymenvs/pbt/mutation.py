# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import random


def mutate_float(x, change_min=1.1, change_max=1.5):
    perturb_amount = random.uniform(change_min, change_max)

    # mutation direction
    new_value = x / perturb_amount if random.random() < 0.5 else x * perturb_amount
    return new_value


def mutate_float_min_1(x, **kwargs):
    new_value = mutate_float(x, **kwargs)
    new_value = max(1.0, new_value)
    return new_value


def mutate_eps_clip(x, **kwargs):
    new_value = mutate_float(x, **kwargs)
    new_value = max(0.01, new_value)
    new_value = min(0.3, new_value)
    return new_value


def mutate_mini_epochs(x, **kwargs):
    change_amount = 1
    new_value = x + change_amount if random.random() < 0.5 else x - change_amount
    new_value = max(1, new_value)
    new_value = min(8, new_value)
    return new_value


def mutate_discount(x, **kwargs):
    """Special mutation func for parameters such as gamma (discount factor)."""
    inv_x = 1.0 - x
    # very conservative, large changes in gamma can lead to very different critic estimates
    new_inv_x = mutate_float(inv_x, change_min=1.1, change_max=1.2)
    new_value = 1.0 - new_inv_x
    return new_value


def get_mutation_func(mutation_func_name):
    try:
        func = eval(mutation_func_name)
    except Exception as exc:
        print(f'Exception {exc} while trying to find the mutation func {mutation_func_name}.')
        raise Exception(f'Could not find mutation func {mutation_func_name}')

    return func


def mutate(params, mutations, mutation_rate, pbt_change_min, pbt_change_max):
    mutated_params = copy.deepcopy(params)

    for param, param_value in params.items():
        # toss a coin whether we perturb the parameter at all
        if random.random() > mutation_rate:
            continue

        mutation_func_name = mutations[param]
        mutation_func = get_mutation_func(mutation_func_name)

        mutated_value = mutation_func(param_value, change_min=pbt_change_min, change_max=pbt_change_max)
        mutated_params[param] = mutated_value

        print(f'Param {param} mutated to value {mutated_value}')

    return mutated_params
