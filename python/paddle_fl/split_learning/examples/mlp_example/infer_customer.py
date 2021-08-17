from paddle.fluid.dygraph.base import to_variable
from paddle.static import InputSpec
import paddle
import numpy as np
import yaml
import logging

from paddle_fl.split_learning.core import CustomerExecutor
import utils

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M', 
        level=logging.INFO)

_LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    paddle.enable_static()
    exe = CustomerExecutor(["0.0.0.0:7858"])
    feed_target_names, fetch_targets = \
            exe.load_inference_model("split/customer_infer")

    for i, item in enumerate(utils.data_iter("data/input.json")):
        uid, _, x2, _ = item
        fetch_vars = exe.run(
                usr_key=uid[0],
                feed={"x2": x2},
                fetch_targets=fetch_targets)
        print("fetch_vars: {}".format(np.array(fetch_vars[0])))
