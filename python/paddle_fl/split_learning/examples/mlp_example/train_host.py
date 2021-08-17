import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.fluid.dygraph.base import to_variable
import paddle
import numpy as np
import logging
from paddle_fl.split_learning.core import HostExecutor, LayerBase
import utils
from utils import SimpleLookupTable, SimpleReader
    
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
        self.embed1 = paddle.nn.Embedding(
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

    @paddle.jit.to_static(input_spec=[
        InputSpec(shape=[None, 12], dtype='int', name='x1')])
    def forward(self, x1):
        self.embed_x1 = self.embed1(x1)
        self.embed_x1 = paddle.reshape(
                self.embed_x1, [-1, 1, self.input_len, self.embed_dim])
        self.pool_x1 = self.pool(self.embed_x1)
        self.pool_x1 = paddle.reshape(
                self.pool_x1, [-1, self.input_len])
        self.fc1_x1 = self.fc1(self.pool_x1)
        return self.fc1_x1


if __name__ == "__main__":
    paddle.disable_static()
    layer = MLPLayer()
    optimizer = paddle.optimizer.SGD(
            learning_rate=0.05,
            parameters=layer.parameters())
    
    exe = HostExecutor(
            table=SimpleLookupTable("data/input.json"), 
            reader=SimpleReader())
    exe.init(
            layer=layer, 
            optimizer=optimizer, 
            tensor_names_to_customer=["fc1_from_host"], # 与 layer.forward 返回值一一对应
            input_specs=[InputSpec(shape=[None, 12], dtype='int', name='x1')])

    # --------------- load params -----------------
    # exe.load_persistables("split/host")
    
    exe.start(7858)
