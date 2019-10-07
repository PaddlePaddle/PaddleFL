from paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.core.master.fl_job import FLRunTimeJob
from paddle_fl.reader.gru4rec_reader import Gru4rec_Reader
import paddle.fluid as fluid
import numpy as np
import sys
import os
import logging
logging.basicConfig(filename="test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

trainer_id = int(sys.argv[1]) # trainer id for each guest
place = fluid.CPUPlace()
train_file_dir = "mid_data/node1/0/"
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.start()

r = Gru4rec_Reader()
train_reader = r.reader(train_file_dir, place)

step_i = 0
while not trainer.stop():
    step_i += 1
    print("batch %d start train" % (step_i))
    for data in train_reader():
        print(data)
        trainer.run(feed=data,
                    fetch=[])
