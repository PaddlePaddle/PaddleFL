import numpy as np
from paddle.vision import datasets, transforms
from utils.util import get_root_path
import os

def get_pt_dataset():
    data_dir = os.path.join(get_root_path(), "data", "cifar100")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=transform_train)

    return train_dataset

if __name__ == '__main__':
    np.random.seed(777)

