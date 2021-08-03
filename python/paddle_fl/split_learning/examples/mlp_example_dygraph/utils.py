import numpy as np
import yaml

from core.table.table_base import TableBase
from core.reader.reader_base import ReaderBase
    

class SimpleLookupTable(TableBase):

    def __init__(self):
        import data_iter
        self.table = []
        for item in data_iter.iter():
            x1, _,  _ = item
            self.table.append(x1)
    
    def _get_value(self, idx):
        return self.table[int(idx)]


class SimpleReader(ReaderBase):
    
    def __init__(self):
        pass
    
    def parse(self, db_value):
        x = db_value
        return {"x1": x}
