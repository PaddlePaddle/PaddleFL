import paddle.fluid as fluid
import numpy as np
import yaml
import logging
import time

from core.static import CustomerExecutor

import network
import utils

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        level=logging.INFO)


if __name__ == "__main__":
    input_from_host, _, label, prediction, cost = network.net()
    exe = CustomerExecutor(
            endpoints=["0.0.0.0:7858"],
            place=fluid.CPUPlace())
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    exe.load_program_from_full_network(
            startup_program=startup_program,
            main_program=main_program)

    for i, item in enumerate(utils.data_iter("../data/input.json")):
        uid, _, x2, label = item
        loss = exe.run(
                usr_key=uid[0],
                feed={
                    "Customer|x2": x2, 
                    "Customer|label": label},
                fetch_list=[cost.name])
        print("loss: {}".format(np.array(loss)))

    if exe.save_persistables(
            "split_program/customer_vars",
            "split_program/host_vars"):
        _LOGGER.info("Succ save vars")

    if exe.save_inference_model(
            "split_program/customer_infer",
            "split_program/host_infer",
            ["Host|x1", "Customer|x2"],
            [prediction.name]):
        _LOGGER.info("Succ save infer model")
