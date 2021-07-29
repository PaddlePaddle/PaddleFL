from paddle.fluid.dygraph.base import to_variable
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
        level=logging.DEBUG)

_LOGGER = logging.getLogger(__name__)

class MLPLayer(LayerBase):
    """
    whole: 
    x1 -> emb -> pool -> fc1 \
                               concat -> fc2 -> softmax
    x2 -> emb -> pool -> fc1 /
    
    host part: 
    x1 -> emb -> pool -> fc1
    
    customer part:  
                      x1_fc1 \
                               concat -> fc2 -> softmax
    x2 -> emb -> pool -> fc1 /
    """
    def __init__(self):
        super(MLPLayer, self).__init__()
        self.input_len = 12
        self.embed_dim = 11
        self.embed2 = paddle.nn.Embedding(
                num_embeddings=100,
                embedding_dim=self.embed_dim,
                weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.1)))
        self.pool = paddle.nn.MaxPool2D(
                kernel_size=[1, self.embed_dim])
        self.fc1 = paddle.nn.Linear(
                in_features=12, 
                out_features=10,
                weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.1)))
        self.fc2 = paddle.nn.Linear(
                in_features=20,
                out_features=2,
                weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.1)))
        self.softmax = paddle.nn.Softmax()

    def forward(self, inputs):
        # return loss
        self.embed_x2 = self.embed2(inputs["x2"])
        self.embed_x2 = paddle.reshape(
                self.embed_x2, [-1, 1, self.input_len, self.embed_dim])
        self.pool_x2 = self.pool(self.embed_x2)
        self.pool_x2 = paddle.reshape(
                self.pool_x2, [-1, self.input_len])
        self.fc1_x2 = self.fc1(self.pool_x2)

        self.concat_var = paddle.concat(
                x=[inputs["fc1"], self.fc1_x2], axis=-1)

        self.fc2_var = self.fc2(self.concat_var)
        self.predict = self.softmax(self.fc2_var)

        loss = paddle.nn.functional.cross_entropy(self.predict, inputs["label"])
        loss = paddle.mean(loss)
        return loss

    def get_fetch_vars(self):
        fetch_vars = {
            "predict": self.predict
        }
        return fetch_vars


if __name__ == "__main__":
    paddle.disable_static()
    layer = MLPLayer()
    optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=layer.parameters())
    common_vars = {
        "in": ["fc1"],
        "out": ["fc1@GRAD"],
    }

    exe = CustomerExecutor(["0.0.0.0:7858"])
    exe.load_layer_handler(layer, optimizer, common_vars)

    for i, item in enumerate(data_iter.iter()):
        _, x2, label = item
        x2_var = to_variable(x2)
        label_var = to_variable(label)

        fetch_vars = exe.run(
                usr_key=str(i),
                feed={
                    "x2": x2_var,
                    "label": label_var})
        print("fetch_vars: {}".format(fetch_vars))
