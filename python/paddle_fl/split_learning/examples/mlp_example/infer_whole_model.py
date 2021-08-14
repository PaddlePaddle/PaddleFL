import paddle
import numpy as np
from paddle.fluid.dygraph.base import to_variable
from paddle.static import InputSpec

import grpc
import yaml

import utils

if __name__ == "__main__":
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    inference_program, feed_target_names, fetch_targets = \
            paddle.static.load_inference_model(
                    path_prefix="whole/static",
                    executor=exe)

    for i, item in enumerate(utils.data_iter("data/input.json")):
        uid, x1, x2, label = item
        feed = {"x1": x1, "x2": x2}
        fetch_vars = exe.run(
                program=inference_program,
                feed=feed,
                fetch_list=fetch_targets,
                return_numpy=False)
        print(np.array(fetch_vars))
