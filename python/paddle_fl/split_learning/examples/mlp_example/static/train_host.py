import paddle.fluid as fluid
import numpy as np
import yaml
import logging
import json
from paddle_fl.split_learning.core.static import HostExecutor
import network
import utils

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        level=logging.INFO)

_LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    host_input, consumer_input, label, prediction, cost = network.net()
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    place = fluid.CPUPlace()
    exe = HostExecutor(
            place=fluid.CPUPlace(), 
            table=utils.SimpleLookupTable("../data/input.json"), 
            reader=utils.SimpleReader())
    exe.load_program_from_full_network(
            startup_program=startup_program,
            main_program=main_program)
    exe.start(7858)
