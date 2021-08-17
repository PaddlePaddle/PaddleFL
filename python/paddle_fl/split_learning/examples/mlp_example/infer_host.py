import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.fluid.dygraph.base import to_variable
import paddle
import numpy as np
import yaml
import logging
from paddle_fl.split_learning.core import HostExecutor
from utils import SimpleLookupTable, SimpleReader
    
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    paddle.enable_static()

    exe = HostExecutor(
            table=SimpleLookupTable("data/input.json"), 
            reader=SimpleReader())
    exe.load_inference_model("split/host_infer")
    exe.start(7858)
