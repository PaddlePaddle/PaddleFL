import paddle.fluid as fluid
import numpy as np
import yaml
import logging
import time

from paddle_fl.split_learning import CustomerExecutor
import network

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        level=logging.DEBUG)


if __name__ == "__main__":
    host_input, label, prediction, cost = network.net()
    place = fluid.CPUPlace()
    exe = CustomerExecutor(
            endpoints=["0.0.0.0:7858"],
            place=place)
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    exe.load_program_from_full_network(
            startup_program=startup_program,
            main_program=main_program)

    customer_input = np.random.randint(2, size=(1, 3)).astype('float32')
    label_data = np.random.randint(2, size=(1, 1)).astype('int64')
    for ei in range(3):
        md5  = "0"
        loss = exe.run(
                usr_key=md5,
                feed={"Customer|input": customer_input, "Customer|label": label_data},
                    fetch_list=[cost.name])
        print("[{}] loss: {}".format(ei, np.array(loss)))

    if exe.save_persistables(
            "split_program/customer_vars",
            "split_program/host_vars"):
        _LOGGER.info("Succ save vars")

    if exe.save_inference_model(
            "split_program/customer_infer",
            "split_program/host_infer",
            ["Host|input"],
            [prediction.name]):
        _LOGGER.info("Succ save infer model")
