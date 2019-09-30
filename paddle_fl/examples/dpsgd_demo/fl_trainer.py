from paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy as np
import sys

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
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.start()

step_i = 0
while not trainer.stop():
    step_i += 1
    print("batch %d start train" % (step_i))
    for data in reader():
        trainer.run(feed=data, fetch=[])
    if step_i % 100 == 0:
        trainer.save_inference_program(output_folder)
