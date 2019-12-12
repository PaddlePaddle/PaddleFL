from paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy
import sys
import logging
import paddle
import paddle.fluid as fluid
import time
import datetime
import math
import hashlib
import hmac

logging.basicConfig(filename="log/test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger("FLTrainer")

BATCH_SIZE = 64

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)
test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

trainer_num = 2
trainer_id = int(sys.argv[1]) # trainer id for each guest

job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.trainer_id = trainer_id
trainer.trainer_num = trainer_num
trainer.key_dir = "./keys/"
trainer.start()

output_folder = "fl_model"
epoch_id = 0
step_i = 0

inputs = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='y', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[inputs, label], place=fluid.CPUPlace())

# for test
test_program = trainer._main_program.clone(for_test=True)

def train_test(train_test_program,
                   train_test_feed, train_test_reader):
    acc_set = []
    avg_loss_set = []
    for test_data in train_test_reader():
        acc_np, avg_loss_np = trainer.exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=["accuracy_0.tmp_0", "mean_0.tmp_0"])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    return avg_loss_val_mean, acc_val_mean
# for test

while not trainer.stop():
    epoch_id += 1
    print("epoch %d start train" % (epoch_id))

    for data in train_reader():
        step_i += 1
        trainer.step_id = step_i
        accuracy, = trainer.run(feed=feeder.feed(data),
            fetch=["accuracy_0.tmp_0"])
        if step_i % 100 == 0:
            print("Epoch: {0}, step: {1}, accuracy: {2}".format(epoch_id, step_i, accuracy[0]))

    avg_loss_val, acc_val = train_test(train_test_program=test_program,
                                       train_test_reader=test_reader,
                                       train_test_feed=feeder)
    print("Test with Epoch %d, avg_cost: %s, acc: %s" %(epoch_id, avg_loss_val, acc_val))

    if epoch_id > 40:
        break
    if step_i % 100 == 0:
        trainer.save_inference_program(output_folder)
