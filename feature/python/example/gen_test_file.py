# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
This modulo gen test file 
"""

import csv
import numpy as np
import random

def gen_simple_file(file_name):
    """
    gen simple test file
    """
    sample = [[1, 0, 0]] * 50 
    sample.extend([[0, 0, 0]] * 200)
    sample.extend([[1, 1, 1]] * 20)
    sample.extend([[0, 1, 1]] * 200)
    sample.extend([[1, 2, 2]] * 5)
    sample.extend([[0, 2, 2]] * 200)
    sample.extend([[1, 3, 3]] * 15)
    sample.extend([[0, 3, 3]] * 200)
    sample.extend([[1, 4, 4]] * 10)
    sample.extend([[0, 4, 4]] * 200)
    random.shuffle(sample)
    sample = np.array(sample)
    np.savetxt(file_name, sample, fmt='%d', delimiter=',')


def gen_zero_test_file(file_name):
    """
    gen zero test file
    """
    sample = [[1, 0, 0]] * 50 
    #sample.extend([[0, 0, 0]] * 200)
    #sample.extend([[1, 1, 1]] * 20)
    sample.extend([[0, 1, 1]] * 200)
    sample.extend([[1, 2, 2]] * 5)
    sample.extend([[0, 2, 2]] * 200)
    sample.extend([[1, 3, 3]] * 15)
    sample.extend([[0, 3, 3]] * 200)
    sample.extend([[1, 4, 4]] * 10)
    sample.extend([[0, 4, 4]] * 200)
    random.shuffle(sample)
    sample = np.array(sample)
    np.savetxt(file_name, sample, fmt='%d', delimiter=',')


def gen_bench_file(file_name):
    """
    gen bench file 
    sample_size = 200000
    feature_size = 3000
    """
    sample = [[1] + [0] * 3000] * 500 * 20
    sample.extend([[0] + [0] * 3000] * 2000 * 20)
    sample.extend([[1] + [1] * 3000] * 200 * 20)
    sample.extend([[0] + [1] * 3000] * 2000 * 20)
    sample.extend([[1] + [2] * 3000] * 50 * 20)
    sample.extend([[0] + [2] * 3000] * 2000 * 20)
    sample.extend([[1] + [3] * 3000] * 150 * 20)
    sample.extend([[0] + [3] * 3000] * 2000 * 20)
    sample.extend([[1] + [4] * 3000] * 100 * 20)
    sample.extend([[0] + [4] * 3000] * 2000 * 20)
    random.shuffle(sample)
    sample = np.array(sample)
    np.savetxt(file_name, sample, fmt='%d', delimiter=',')


def read_file(file_name):
    """
    read file
    """
    sample = np.loadtxt(file_name, dtype=np.int32, delimiter=',')
    sample = sample.tolist()
    labels = [val[:1] for val in sample]
    features = [val[1:] for val in sample]
    return labels, features

if __name__ == '__main__':
    gen_simple_file("test_data.txt")
    #gen_zero_test_file("test_data_zero.txt")
    #gen_bench_file("test_data_bench.txt")

