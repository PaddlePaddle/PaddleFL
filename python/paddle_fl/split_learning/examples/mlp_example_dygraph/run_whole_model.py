import paddle
import numpy as np
from paddle.fluid.dygraph.base import to_variable
import grpc
import yaml

import data_iter

class MLP(paddle.nn.Layer):

    def __init__(self):
        super(MLP, self).__init__()
        self.input_len = 12
        self.embed_dim = 11
        self.embed1 = paddle.nn.Embedding(
                num_embeddings=100,
                embedding_dim=self.embed_dim,
                weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.1)))
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
        self.embed_x1 = self.embed1(inputs["x1"])
        self.embed_x1 = paddle.reshape(
                self.embed_x1, [-1, 1, self.input_len, self.embed_dim])
        self.pool_x1 = self.pool(self.embed_x1)
        self.pool_x1 = paddle.reshape(
                self.pool_x1, [-1, self.input_len])
        self.fc1_x1 = self.fc1(self.pool_x1)

        self.embed_x2 = self.embed2(inputs["x2"])
        self.embed_x2 = paddle.reshape(
                self.embed_x2, [-1, 1, self.input_len, self.embed_dim])
        self.pool_x2 = self.pool(self.embed_x2)
        self.pool_x2 = paddle.reshape(
                self.pool_x2, [-1, self.input_len])
        self.fc1_x2 = self.fc1(self.pool_x2)

        self.concat_var = paddle.concat(
                x=[self.fc1_x1, self.fc1_x2], axis=-1)

        self.fc2_var = self.fc2(self.concat_var)
        self.predict = self.softmax(self.fc2_var)
        return self.predict


if __name__ == "__main__":
    layer = MLP()

    optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=layer.parameters())

    for i, item in enumerate(data_iter.iter()):
        x1, x2, label = item
        x1_var = to_variable(x1)
        x2_var = to_variable(x2)
        label_var = to_variable(label)

        predict = layer({"x1": x1_var, "x2": x2_var})
        print(predict)
        cost = paddle.nn.functional.cross_entropy(predict, label_var)
        cost = paddle.mean(cost)

        cost.backward()
        optimizer.step()
        layer.clear_gradients()
