# -*- coding: utf-8 -*-
# Copyright (c) Percy.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
dprl.utils
~~~~~~~~
Useful tools for DPRL
"""

import numpy as np
from pathlib import Path

from .ldp import Numeric


def gen_epsilon_dist(epsilon_budget, num, high, low):
    high_num = (epsilon_budget - low * num) / (high - low)
    high_num = int(high_num)
    low_num = num - high_num
    return high_num, low_num


def gen_epsilon_list(s_input, high, low):
    epsilon_list = np.zeros(len(s_input))
    for i, select in enumerate(s_input):
        epsilon_list[i] = low if select == 0 else high
    return epsilon_list


def process_train_data(x, y, epsilon_list):
    for i in range(len(y)):
        tp = np.append(x[i], y[i])
        processed_tp = Numeric.wang_multi_dim_method(tp, epsilon_list[i], 'pm')

        if processed_tp[-1] <= 1:
            processed_tp[-1] = 0
        else:
            processed_tp[-1] = 2

        x[i] = np.asarray(processed_tp[:-1])
        y[i] = processed_tp[-1]

    return x, y


def save_npy(file_name, arr):
    check_folder('res')
    if isinstance(arr, np.ndarray):
        np.save(str(Path.cwd().joinpath('res').joinpath(file_name)), arr)


def load_npy(file_name):
    check_folder('res')
    arr = np.load(str(Path.cwd().joinpath('res').joinpath(file_name)))
    if isinstance(arr, np.ndarray):
        return arr


def check_folder(folder_name):
    if not Path(folder_name).exists():
        prefix = Path.cwd()
        Path.mkdir(prefix.joinpath(folder_name))


