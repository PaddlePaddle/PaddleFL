from paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy as np
import sys
import logging
import time
logging.basicConfig(filename="test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)


def reader():
    for i in range(1000):
        data_dict = {}
        for i in range(3):
            data_dict[str(i)] = np.random.rand(1, 5).astype('float32')
        data_dict["label"] = np.random.randint(2, size=(1, 1)).astype('int64')
        yield data_dict

trainer_id = int(sys.argv[1]) # trainer id for each guest
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000+trainer_id)
trainer.start()
print(trainer._scheduler_ep, trainer._current_ep)
output_folder = "fl_model"
step_i = 0
while not trainer.stop():
    print("batch %d start train" % (step_i))
    train_step = 0
    for data in reader():
        trainer.run(feed=data, fetch=[])
        train_step += 1
        if train_step == trainer._step:
            break
    step_i += 1
    if step_i % 100 == 0:
        trainer.save_inference_program(output_folder)
    
