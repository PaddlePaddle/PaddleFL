import numpy as np
import yaml
import json
import paddle.fluid as fluid
from paddle_fl.split_learning.core import TableBase, ReaderBase
    

def data_iter(filename):
    batch_size = 1
    with open(filename) as f:
        batch_uid = []
        batch_title = []
        batch_click = []
        batch_action = []
        for line in f:
            info = json.loads(line.strip('\n'))
            uid = info["user_info"]["uid"]
            title = [int(x) for x in info["item_info"][0]["title"]]
            if len(title) < 12:
                title += [0 for _ in range(12 - len(title))]
            action = [int(x) for x in info["item_info"][0]["action"]]
            if len(action) < 12:
                action += [0 for _ in range(12 - len(action))]
            click = [int(info["item_info"][0]["click"])]
            batch_uid.append(uid)
            batch_title.append(title)
            batch_click.append(click)
            batch_action.append(action)
            if len(batch_title) == batch_size:
                title_data = np.asarray(batch_title, dtype="int64")
                title_data.reshape([batch_size, 1, 1, len(title)])
                action_data = np.asarray(batch_action, dtype="int64")
                action_data.reshape([batch_size, 1, 1, len(action)])
                click_data = np.asarray(batch_click, dtype="int64")
                yield (batch_uid, create_lod_tensor(title_data), create_lod_tensor(action_data), click_data)
                batch_uid = []
                batch_title = []
                batch_click = []
                batch_action = []

def create_lod_tensor(data, place=fluid.CPUPlace()):
    data = np.asarray(data, dtype="int64")
    shapes = [[len(c) for c in data]]
    return fluid.create_lod_tensor(data.reshape(-1, 1), shapes, place)


class SimpleLookupTable(TableBase):

    def __init__(self, filename):
        self.table = {}
        for item in data_iter(filename):
            uid, x1, _,  _ = item
            self.table[uid[0]] = x1
    
    def _get_value(self, uid):
        return self.table[uid]


class SimpleReader(ReaderBase):
    
    def __init__(self):
        pass
    
    def parse(self, table_value):
        x = table_value
        return {"Host|x1": x}
