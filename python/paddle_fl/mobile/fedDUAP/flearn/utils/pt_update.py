import paddle
from paddle.io import Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Paddle Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return paddle.Tensor(image), paddle.Tensor(label)
