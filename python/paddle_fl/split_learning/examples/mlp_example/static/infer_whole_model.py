import paddle.fluid as fluid
import numpy as np
import grpc
import yaml
import paddle

import network
import utils

if __name__ == "__main__":
    paddle.enable_static()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_program, feed_target_names, fetch_targets = \
            fluid.io.load_inference_model(
                    dirname="whole_program/inference", 
                    executor=exe)

    for batch_id, data in enumerate(utils.data_iter("../data/input.json")):
        _, x1, x2, _ = data
        data = {
            "Host|x1": x1,
            "Customer|x2": x2,
        }
        results = exe.run(
                program=inference_program,
                feed=data,
                fetch_list=fetch_targets)
        print("result: {}".format(np.array(results)))
