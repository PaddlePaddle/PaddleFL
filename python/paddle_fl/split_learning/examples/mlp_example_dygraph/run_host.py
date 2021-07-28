import paddle.fluid as fluid
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
    MLP: x -> emb -> pool -> fc1 -> fc2 -> label

    host part: x -> emb -> pool -> fc1
    customer part: fc1 -> fc2 -> label
    """
    def __init__(self):
        super(MLPLayer, self).__init__()
        self.embed_dim = 11
        self.embed = fluid.dygraph.Embedding(
                size=[100, self.embed_dim],
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))
        self.pool = fluid.dygraph.Pool2D(
                pool_type='max',
                global_pooling=True)
        self.fc1 = fluid.dygraph.nn.Linear(
                input_dim=12, 
                output_dim=10,
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    def forward(self, inputs):
        self.embed_var = self.embed(inputs["x"])
        self.embed_var = fluid.layers.reshape(
                self.embed_var, 
                [-1, 12, 1, self.embed_dim])
        self.pool_var = self.pool(self.embed_var)
        self.pool_var = fluid.layers.reshape(
                self.pool_var, 
                [-1, 12])
        self.fc1_var = self.fc1(self.pool_var)

    def get_fetch_vars(self):
        fetch_vars = {
            "fc1": self.fc1_var
        }
        return fetch_vars


class LookupTable(TableBase):

    def __init__(self):
        import data_iter
        self.table = []
        for item in data_iter.iter():
            slot, _ = item
            self.table.append(slot)
    
    def _get_value(self, idx):
        return self.table[int(idx)]


class Reader(ReaderBase):
    
    def __init__(self):
        pass
    
    def parse(self, db_value):
        x = fluid.dygraph.to_variable(db_value)
        return {"x": x}



if __name__ == "__main__":
    place = fluid.CPUPlace()
    fluid.enable_imperative(place)
    layer = MLPLayer()
    optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=0.01,
            parameter_list=layer.parameters())
    common_vars = {
        "in": ["fc1@GRAD"],
        "out": ["fc1"],
    }

    table = LookupTable()
    reader = Reader()
    
    exe = HostExecutor(table, reader)
    exe.load_layer_handler(layer, optimizer, common_vars)
    exe.start(7858)
