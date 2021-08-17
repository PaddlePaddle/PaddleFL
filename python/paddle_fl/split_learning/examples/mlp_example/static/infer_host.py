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

if __name__ == "__main__":
    place = fluid.CPUPlace()
    exe = HostExecutor(
            place=fluid.CPUPlace(), 
            table=utils.SimpleLookupTable("../data/input.json"),
            reader=utils.SimpleReader())
    exe.load_inference_model("split_program/host_infer")

    exe.start(7858)
