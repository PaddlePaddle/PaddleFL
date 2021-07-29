import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import paddle
import numpy as np
import yaml
import logging

from core.dygraph.layer_handler import HostLayerHandler, LayerBase
from core.dygraph import HostExecutor
from core.table.table_base import TableBase
from core.reader.reader_base import ReaderBase
    
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
        self.embed1 = paddle.nn.Embedding(
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

    def forward(self, inputs):
        self.embed_x1 = self.embed1(inputs["x1"])
        self.embed_x1 = paddle.reshape(
                self.embed_x1, [-1, 1, self.input_len, self.embed_dim])
        self.pool_x1 = self.pool(self.embed_x1)
        self.pool_x1 = paddle.reshape(
                self.pool_x1, [-1, self.input_len])
        self.fc1_x1 = self.fc1(self.pool_x1)
        return None

    def get_fetch_vars(self):
        fetch_vars = {
            "fc1": self.fc1_x1
        }
        return fetch_vars


class LookupTable(TableBase):

    def __init__(self):
        import data_iter
        self.table = []
        for item in data_iter.iter():
            x1, _,  _ = item
            self.table.append(x1)
    
    def _get_value(self, idx):
        return self.table[int(idx)]


class Reader(ReaderBase):
    
    def __init__(self):
        pass
    
    def parse(self, db_value):
        x = to_variable(db_value)
        return {"x1": x}


if __name__ == "__main__":
    paddle.disable_static()
    layer = MLPLayer()
    optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=layer.parameters())
    common_vars = {
        "in": ["fc1@GRAD"],
        "out": ["fc1"],
    }

    table = LookupTable()
    reader = Reader()
    
    exe = HostExecutor(table, reader)
    exe.load_layer_handler(layer, optimizer, common_vars)
    exe.start(7858)
