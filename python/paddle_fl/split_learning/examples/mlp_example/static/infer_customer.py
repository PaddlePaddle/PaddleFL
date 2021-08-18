import paddle.fluid as fluid
import numpy as np
import yaml
import logging
from paddle_fl.split_learning.core.static import CustomerExecutor
import network
import utils

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M', 
        level=logging.INFO)


if __name__ == "__main__":
    exe = CustomerExecutor(
            endpoints=["0.0.0.0:7858"],
            place=fluid.CPUPlace())

    feed_target_names, fetch_targets = \
            exe.load_inference_model("split_program/customer_infer")

    for batch_id, data in enumerate(utils.data_iter("../data/input.json")):
        uid, _, x2, _ = data
        result = exe.run(
                usr_key=uid[0],
                feed={"Customer|x2": x2},
                fetch_list=fetch_targets)
        print("result: {}".format(np.array(result)))
