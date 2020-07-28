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
from paddle_fl.paddle_fl.reader.gru4rec_reader import Gru4rec_Reader
import paddle.fluid as fluid
import numpy as np
import sys
import os
import logging
import time

logging.basicConfig(
    filename="test.log",
    filemode="w",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.DEBUG)

trainer_id = int(sys.argv[1])  # trainer id for each guest
place = fluid.CPUPlace()
train_file_dir = "mid_data/node4/%d/" % trainer_id
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform the scheduler IP to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = fluid.CPUPlace()
trainer.start(place)

r = Gru4rec_Reader()
train_reader = r.reader(train_file_dir, place, batch_size=125)

output_folder = "model_node4"
epoch_i = 0
while not trainer.stop():
    epoch_i += 1
    train_step = 0
    for data in train_reader():
        #print(np.array(data['src_wordseq']))
        ret_avg_cost = trainer.run(feed=data, fetch=["mean_0.tmp_0"])
        train_step += 1
        if train_step == trainer._step:
            break
        avg_ppl = np.exp(ret_avg_cost[0])
        newest_ppl = np.mean(avg_ppl)
        print("{} Epoch {} start train, train_step {}, ppl {}".format (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch_i, train_step, newest_ppl))
    save_dir = (output_folder + "/epoch_%d") % epoch_i
    if trainer_id == 0:
        print("start save")
        trainer.save_inference_program(save_dir)
    if epoch_i >= 5:
        break
