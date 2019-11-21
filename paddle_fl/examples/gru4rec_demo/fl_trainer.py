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
train_file_dir = "mid_data/node4/%d/" % trainer_id
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000+trainer_id)
trainer.start()

r = Gru4rec_Reader()
train_reader = r.reader(train_file_dir, place, batch_size = 125)

output_folder = "model_node4"
step_i = 0
while not trainer.stop():
    step_i += 1
    print("batch %d start train" % (step_i))
    train_step = 0
    for data in train_reader():
        #print(np.array(data['src_wordseq']))
        ret_avg_cost = trainer.run(feed=data,
                    fetch=["mean_0.tmp_0"])
        train_step += 1
	if train_step == trainer._step:
		break
        avg_ppl = np.exp(ret_avg_cost[0])
        newest_ppl = np.mean(avg_ppl)
        print("ppl:%.3f" % (newest_ppl))
    save_dir = (output_folder + "/epoch_%d") % step_i
    if trainer_id == 0:
        print("start save")
        trainer.save_inference_program(save_dir)
    if step_i >= 40:
        break
