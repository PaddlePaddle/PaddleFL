class ReaderBase(object):

    def __init__(self):
        pass

    def parse(self, db_value):
        raise NotImplementedError("Failed to parse db_value")
