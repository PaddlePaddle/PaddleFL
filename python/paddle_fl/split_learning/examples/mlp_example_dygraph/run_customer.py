import paddle.fluid as fluid
import numpy as np
import yaml
import logging

from core.dygraph.layer_handler import CustomerLayerHandler, LayerBase
from core.dygraph import CustomerExecutor
import data_iter

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M', 
        level=logging.DEBUG)

_LOGGER = logging.getLogger(__name__)

class MLPLayer(LayerBase):
    """
    MLP: x -> emb -> pool -> fc1 -> fc2 -> label

    host part: x -> emb -> pool -> fc1
    customer part: fc1 -> fc2 -> label
    """
    def __init__(self):
        super(MLPLayer, self).__init__()
        self.embed_dim = 11
        self.fc2 = fluid.dygraph.nn.Linear(
                input_dim=10,
                output_dim=2,
                act='softmax',
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    def forward(self, inputs):
        # return loss
        self.predict = self.fc2(inputs["fc1"])
        self.loss = fluid.layers.cross_entropy(self.predict, inputs["label"])
        self.loss = fluid.layers.reduce_mean(self.loss)
        return self.loss

    def get_fetch_vars(self):
        fetch_vars = {
            "predict": self.predict
        }
        return fetch_vars


if __name__ == "__main__":
    place = fluid.CPUPlace()
    fluid.enable_imperative(place)
    layer = MLPLayer()
    optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=0.01,
            parameter_list=layer.parameters())
    common_vars = {
        "in": ["fc1"],
        "out": ["fc1@GRAD"],
    }

    exe = CustomerExecutor(["0.0.0.0:7858"])
    exe.load_layer_handler(layer, optimizer, common_vars)

    for i, item in enumerate(data_iter.iter()):
        _, label = item
        label_var = fluid.dygraph.to_variable(label)

        fetch_vars = exe.run(
                usr_key=str(i),
                feed={"label": label_var})
        print("fetch_vars: {}".format(fetch_vars))
