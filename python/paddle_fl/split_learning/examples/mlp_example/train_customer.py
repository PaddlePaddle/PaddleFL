from paddle.fluid.dygraph.base import to_variable
from paddle.static import InputSpec
import paddle
import numpy as np
import yaml
import logging
from paddle_fl.split_learning.core import CustomerExecutor, LayerBase
import utils

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M', 
        level=logging.INFO)


class MLPLayer(LayerBase):
    """
    whole: 
    x1 -> emb -> pool -> fc1 \
                               concat -> fc2 -> softmax
    x2 -> emb -> pool -> fc1 /
    

    host part: 
    x1 -> emb -> pool -> fc1 (to customer)
    

    customer part:  
             fc1 (from host) \
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
                        initializer=paddle.nn.initializer.Constant(value=0.5)))
        self.pool = paddle.nn.MaxPool2D(
                kernel_size=[1, self.embed_dim])
        self.fc1 = paddle.nn.Linear(
                in_features=12, 
                out_features=10,
                weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.5)))
        self.fc2 = paddle.nn.Linear(
                in_features=20,
                out_features=2,
                weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.5)))
        self.softmax = paddle.nn.Softmax()

    @paddle.jit.to_static(input_spec=[
        InputSpec(shape=[None, 10], dtype='float32', name='fc1_from_host'),
        InputSpec(shape=[None, 12], dtype='int', name='x2')])
    def forward(self, fc1_from_host, x2):
        self.embed_x2 = self.embed2(x2)
        self.embed_x2 = paddle.reshape(
                self.embed_x2, [-1, 1, self.input_len, self.embed_dim])
        self.pool_x2 = self.pool(self.embed_x2)
        self.pool_x2 = paddle.reshape(
                self.pool_x2, [-1, self.input_len])
        self.fc1_x2 = self.fc1(self.pool_x2)

        self.concat_var = paddle.concat(
                x=[fc1_from_host, self.fc1_x2], axis=-1)

        self.fc2_var = self.fc2(self.concat_var)
        self.predict = self.softmax(self.fc2_var)
        return self.predict

    def get_loss(self, fetch_list, label):
        loss = paddle.nn.functional.cross_entropy(fetch_list[0], label)
        loss = paddle.mean(loss)
        return loss


if __name__ == "__main__":
    paddle.disable_static()
    layer = MLPLayer()
    optimizer = paddle.optimizer.SGD(
            learning_rate=0.05,
            parameters=layer.parameters())

    exe = CustomerExecutor(["0.0.0.0:7858"])
    exe.init(
            layer=layer, 
            optimizer=optimizer, 
            tensor_names_from_host=["fc1_from_host"],
            input_specs=[
                InputSpec(shape=[None, 10], dtype='float32', name='fc1_from_host'),
                InputSpec(shape=[None, 12], dtype='int', name='x2')])

    # --------------- load params -----------------
    # exe.load_persistables("split/customer")

    for i, item in enumerate(utils.data_iter("data/input.json")):
        uid, _, x2, label = item
        x2_var = to_variable(x2)
        label_var = to_variable(label)

        fetch_vars, loss = exe.run(
                usr_key=uid[0],
                feed={"x2": x2_var},
                label=label_var)
        print("predict: {}, loss: {}".format(
            fetch_vars[0].numpy(), loss.numpy()))
    
    # --------------- save params -----------------
    # exe.save_persistables(
    #          local_path="split/customer", 
    #          remote_path="split/host")
    
    # --------------- save inference model -----------------
    exe.save_inference_model(
            local_path="split/customer_infer",
            remote_path="split/host_infer")
