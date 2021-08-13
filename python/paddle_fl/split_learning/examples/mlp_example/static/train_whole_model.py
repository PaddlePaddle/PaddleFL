import paddle.fluid as fluid
import numpy as np
import time
import grpc
import json
import yaml
import paddle

import network
import utils

if __name__ == "__main__":
    paddle.enable_static()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    slot1, slot2, label, prediction, cost = network.net()
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()
 
    #startup_program, main_program = split_nn.util.load_whole_program("whole_program")
    exe.run(startup_program)
    
    """
    fluid.io.load_persistables(
            executor=exe, 
            dirname="whole_program/persistables_vars",
            main_program=main_program)
    """

    for i, item in enumerate(utils.data_iter("../data/input.json")):
        uid, x1, x2, label = item
        data = {
            "Host|x1": x1,
            "Customer|x2": x2,
            "Customer|label": label,
        }
        loss = exe.run(
            program=main_program,
            feed=data,
            fetch_list=[cost.name])
        print("loss: {}".format(np.array(loss)))

    fluid.io.save_persistables(
            executor=exe, 
            dirname="whole_program/persistables_vars",
            main_program=main_program)

    fluid.io.save_inference_model(
            dirname="whole_program/inference", 
            feeded_var_names=['Host|x1', "Customer|x2"], 
            target_vars=[prediction], 
            executor=exe,
            main_program=main_program)
