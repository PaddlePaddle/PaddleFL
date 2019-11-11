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
from diffiehellman.diffiehellman import DiffieHellman
# from enum import Enum
# Party = Enum('Party', ('alice', 'bob'))

logging.basicConfig(filename="test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)


train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
    batch_size=16)

trainer_num = 2
trainer_id = int(sys.argv[1]) # trainer id for each guest

job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.start()

output_folder = "fl_model"
epoch_id = 0
step_i = 0
while not trainer.stop():
    epoch_id += 1
    print("epoch %d start train" % (epoch_id))
    starttime = datetime.datetime.now()

    # prepare the aggregated parameters
    param_name_list = []
    param_name_list.append("fc_0.b_0.opti.trainer_" + str(trainer_id))
    param_name_list.append("fc_0.b_0.opti.trainer_" + str(trainer_id))
    
    inputs = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='y', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[inputs, label], place=fluid.CPUPlace())

    scale = pow(10.0, 5)
    # 1. load priv key and other's pub key
    # party_name = Party(trainer_id + 1)
    dh = DiffieHellman(group=15, key_length=256)
    dh.load_private_key(str(trainer_id) + "_priv_key.txt")

    digestmod="SHA256"

    for data in train_reader():
        step_i += 1
        noise = 0.0

        # 2. generate noise        
        secagg_starttime = datetime.datetime.now()
        key = str(step_i).encode("utf-8")
        for i in range(trainer_num):
            if i != trainer_id:
                f = open(str(i) + "_pub_key.txt", "r")
                public_key = int(f.read())
                dh.generate_shared_secret(public_key, echo_return_key=True)
                msg = dh.shared_key.encode("utf-8")
                hex_res1 = hmac.new(key=key, msg=msg, digestmod=digestmod).hexdigest()
                current_noise = int(hex_res1[0:8], 16) / scale
                if i > trainer_id:
                    noise = noise + current_noise
                else:
                    noise = noise - current_noise
        #secagg_endtime = datetime.datetime.now()
        #print("Epoch: {0}, step: {1}".format(epoch_id, step_i))
        #print("secagg time cost: {0}".format(secagg_endtime - secagg_starttime))
       
        if step_i % 100 == 0:
            print("Step: {0}".format(step_i))
        # 3. add noise between training and sending.
        accuracy, = trainer.run(feed=feeder.feed(data), 
            fetch=["top_k_0.tmp_0"], 
            param_name_list=param_name_list, 
            mask=noise)
        #train_endtime = datetime.datetime.now()
        #print("train time cost: {0}".format(train_endtime - secagg_endtime))
     
    print("Epoch: {0}, step: {1}, accuracy: {2}".format(epoch_id, step_i, accuracy[0]))
    endtime = datetime.datetime.now()
    print("time cost: {0}".format(endtime - starttime))

    if epoch_id > 40:
        break
    if step_i % 100 == 0:
        #print("Epoch: {0},loss: {1}".format(step_i, loss_value[0]))
        trainer.save_inference_program(output_folder)
