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

from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy
import sys
import paddle
import paddle.fluid as fluid
import logging
import math
import time

logging.basicConfig(
    filename="test.log",
    filemode="w",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.DEBUG)

trainer_id = int(sys.argv[1])  # trainer id for each guest
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform scheduler IP address to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = fluid.CPUPlace()
trainer.start(place)

test_program = trainer._main_program.clone(for_test=True)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=64)
test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=64)

img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[img, label], place=fluid.CPUPlace())


def train_test(train_test_program, train_test_feed, train_test_reader):
    acc_set = []
    for test_data in train_test_reader():
        acc_np = trainer.exe.run(program=train_test_program,
                                 feed=train_test_feed.feed(test_data),
                                 fetch_list=["accuracy_0.tmp_0"])
        acc_set.append(float(acc_np[0]))
    acc_val_mean = numpy.array(acc_set).mean()
    return acc_val_mean


def compute_privacy_budget(sample_ratio, epsilon, step, delta):
    E = 2 * epsilon * math.sqrt(step * sample_ratio)
    print("({0}, {1})-DP".format(E, delta))


output_folder = "model_node%d" % trainer_id
epoch_id = 0
step = 0
while not trainer.stop():
    epoch_id += 1
    if epoch_id > 10:
        break
    print("{} Epoch {} start train".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch_id))
    for step_id, data in enumerate(train_reader()):
        acc = trainer.run(feeder.feed(data), fetch=["accuracy_0.tmp_0"])
        step += 1
    # print("acc:%.3f" % (acc[0]))

    acc_val = train_test(
        train_test_program=test_program,
        train_test_reader=test_reader,
        train_test_feed=feeder)

    print("Test with epoch %d, accuracy: %s" % (epoch_id, acc_val))
    compute_privacy_budget(
        sample_ratio=0.001, epsilon=0.1, step=step, delta=0.00001)

    save_dir = (output_folder + "/epoch_%d") % epoch_id
    trainer.save_inference_program(output_folder)
