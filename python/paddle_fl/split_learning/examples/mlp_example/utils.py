import numpy as np
import yaml

from core.table.table_base import TableBase
from core.reader.reader_base import ReaderBase
    

class SimpleLookupTable(TableBase):

    def __init__(self):
        import data_iter
        self.table = {}
        for item in data_iter.iter():
            uid, x1, _,  _ = item
            self.table[uid[0]] = x1
    
    def _get_value(self, uid):
        return self.table[uid]


class SimpleReader(ReaderBase):
    
    def __init__(self):
        pass
    
    def parse(self, table_value):
        x = table_value
        return {"x1": x}
