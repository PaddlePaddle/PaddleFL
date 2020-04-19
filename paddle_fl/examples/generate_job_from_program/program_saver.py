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
# limitations under the License.

import os
import json
import paddle.fluid as fluid
#from paddle.fluid.framework import Parameter, Variable

input = fluid.layers.data(name='input', shape=[1, 28, 28], dtype="float32")
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[input, label], place=fluid.CPUPlace())
predict = fluid.layers.fc(input=input, size=10, act='softmax')
sum_cost = fluid.layers.cross_entropy(input=predict, label=label)
accuracy = fluid.layers.accuracy(input=predict, label=label)
avg_cost = fluid.layers.mean(sum_cost, name="loss")
startup_program = fluid.default_startup_program()

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)


def save_program(program_path):
    if not os.path.exists(program_path):
        os.makedirs(program_path)
    main_program_str = fluid.default_main_program().desc.serialize_to_string()
    startup_program_str = fluid.default_startup_program(
    ).desc.serialize_to_string()
    params = fluid.default_main_program().global_block().all_parameters()
    para_info = []
    for pa in params:
        para_info.append(pa.name)
    with open(program_path + '/para_info', 'wb') as fout:
        for item in para_info:
            fout.write("%s\n" % item)
    with open(program_path + '/startup_program', "wb") as fout:
        fout.write(startup_program_str)
    with open(program_path + '/main_program', "wb") as fout:
        fout.write(main_program_str)


program_path = './load_file'
save_program(program_path)
