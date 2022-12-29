# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

import random

file_handle1=open('Input-P0.list',mode='w')
for i in range(10):
    file_handle1.write(str(round(random.uniform(10,20),6)) + '\n')

file_handle1=open('Input-P1.list',mode='w')
for i in range(10):
    file_handle1.write(str(round(random.random(),6)) + '\n')

file_handle1=open('Input-P2.list',mode='w')
for i in range(10):
    file_handle1.write(str(round(random.uniform(1,2),6)) + '\n')
