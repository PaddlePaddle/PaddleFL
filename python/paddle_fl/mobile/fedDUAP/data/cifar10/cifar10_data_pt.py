"""
Get data
"""
import numpy as np
from torchvision import datasets, transforms
from utils.util import get_root_path
import os


def get_pt_dataset():
    """
    Get data
    """
    data_dir = os.path.join(get_root_path(), "data", "cifar10")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=transform_train)

    return train_dataset


if __name__ == '__main__':
    np.random.seed(777)
    get_pt_dataset()
