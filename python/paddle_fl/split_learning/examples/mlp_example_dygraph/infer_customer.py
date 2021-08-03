from paddle.fluid.dygraph.base import to_variable
from paddle.static import InputSpec
import paddle
import numpy as np
import yaml
import logging

from core.dygraph.layer_handler import CustomerLayerHandler, LayerBase
from core.dygraph import CustomerExecutor
import data_iter

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

    for i, item in enumerate(data_iter.iter()):
        _, x2, _ = item
        fetch_vars = exe.run(
                usr_key=str(i),
                feed={"x2": x2},
                fetch_targets=fetch_targets)
        print("fetch_vars: {}".format(np.array(fetch_vars[0])))
