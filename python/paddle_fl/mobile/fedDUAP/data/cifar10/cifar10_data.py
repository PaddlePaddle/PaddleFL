"""
generate cifar iid data
"""
import numpy as np
from utils.util import get_root_path
from data.util import show_data
from paddle.vision import datasets, transforms
import os
import paddle
from paddle.vision.transforms import ToTensor


def cifar_iid(dataset, num_users, num_data, num_share=0):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    targets = []
    for data in dataset:
        targets.append(data[1].item())
    labels = np.array(targets)[:num_data]
    max_value = max(targets)
    dataset.classes = [i for i in range(0, max_value + 1)]

    num_items = int(num_data / num_users)
    dict_users, all_idxs = {}, [i for i in range(num_data)]
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items,
                                         replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))

    # get shared data
    dict_users[num_users] = np.random.choice(range(num_data, len(dataset)), num_share, replace=False)

    # show data distribution
    show_data(dataset, dict_users)
    return dict_users


def cifar_noniid(dataset, num_users, num_data, num_share=0, l=2, l_share=10):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards = num_users * l
    num_imgs = int(num_data / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    targets = []
    for data in dataset:
        targets.append(data[1].item())
    labels = np.array(targets)[:num_data]
    max_value = max(targets)
    dataset.classes = [i for i in range(0, max_value + 1)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, l, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    # get shared data according to l_share categories
    dict_users[num_users] = np.array([])
    idxs = np.arange(num_data, len(dataset))
    labels = np.array(targets)[num_data:]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # count num of each category and begin index
    num_category = np.bincount(idxs_labels[1].flatten())
    idx_category = np.zeros(len(dataset.classes) + 1, dtype=np.int64)
    for i in range(1, len(num_category) + 1):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    num_each = int(num_share / l_share)
    rand_category = np.random.choice(range(0, len(dataset.classes)), l_share, replace=False)
    for category in rand_category:
        choices = idxs[idx_category[category]: idx_category[category + 1]]
        dict_users[num_users] = np.concatenate(
            (dict_users[num_users], np.random.choice(choices, min(num_each, num_category[category]), replace=False)),
            axis=0)

    # show data distribution
    show_data(dataset, dict_users)
    return dict_users


def cifar_noniid_unequal(dataset, num_users, num_data, num_share=0, l=2, l_share=10):
    """
    devices have different nums of data and categories
    :param dataset:
    :param num_users:
    :param num_data:
    :param num_share:
    :param l:
    :return:
    """
    idxs = np.arange(num_data)
    targets = []
    for data in dataset:
        targets.append(data[1].item())
    labels = np.array(targets)[:num_data]
    max_value = max(targets)
    dataset.classes = [i for i in range(0, max_value + 1)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # count num of each category and begin index
    num_category = np.bincount(idxs_labels[1].flatten())
    idx_category = np.zeros(len(dataset.classes), dtype=np.int64)
    for i in range(1, len(num_category)):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    # decide device data num for each device
    num_samples = np.random.lognormal(3, 1, num_users) + 10  # 4, 1.5 for data 1
    num_samples = num_data * num_samples / sum(num_samples)  # normalize

    dict_users = {i: np.array([]) for i in range(num_users)}
    class_per_user = np.ones(num_users) * l
    idx_train = np.zeros(len(dataset.classes), dtype=np.int64)
    for i in range(len(idx_train)):
        idx_train[i] = idx_category[i]
    for user in range(num_users):
        props = np.random.lognormal(1, 1, int(class_per_user[user]))
        props = props / sum(props)
        for j in range(int(class_per_user[user])):
            class_id = (user + j) % len(dataset.classes)
            train_sample_this_class = int(props[j] * num_samples[user]) + 1

            if idx_train[class_id] + train_sample_this_class > num_category[class_id]:
                idx_train[class_id] = idx_category[class_id]

            dict_users[user] = np.concatenate(
                (dict_users[user], idxs[idx_train[class_id]: idx_train[class_id] + train_sample_this_class]), axis=0)

            idx_train[class_id] += train_sample_this_class
    # get shared data
    # dict_users[num_users] = np.random.choice(range(num_data, len(dataset)), num_share, replace=False)

    # get shared data
    dict_users[num_users] = np.array([])
    idxs = np.arange(num_data, len(dataset))
    labels = np.array(targets)[num_data:]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # count num of each category and begin index
    num_category = np.bincount(idxs_labels[1].flatten())
    idx_category = np.zeros(len(dataset.classes) + 1, dtype=np.int64)
    for i in range(1, len(num_category) + 1):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    num_each = int(num_share / l_share)
    rand_category = np.random.choice(range(0, len(dataset.classes)), l_share, replace=False)
    for category in rand_category:
        choices = idxs[idx_category[category]: idx_category[category + 1]]
        dict_users[num_users] = np.concatenate(
            (
                dict_users[num_users], np.random.choice(choices, min(num_each, num_category[category]), replace=False)),
            axis=0)

    # show data distribution
    show_data(dataset, dict_users)
    return dict_users


def get_dataset(num_data=40000, num_users=100, iid=True, unequal=False, l=2, num_share=0, share_l=10):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = os.path.join(get_root_path(), "data", "cifar10")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)

    test_dataset = datasets.Cifar10(mode='test', transform=transform_test)

    # sample training data amongst users
    if iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, num_users, num_data, num_share)
    else:
        # Sample Non-IID user data from Mnist
        if unequal:
            # Chose unequal splits for every user
            user_groups = cifar_noniid_unequal(train_dataset, num_users, num_data, num_share, l, share_l)
            # raise NotImplementedError()
        else:
            # Chose equal splits for every user
            user_groups = cifar_noniid(train_dataset, num_users, num_data, num_share, l, share_l)

    return train_dataset, test_dataset, user_groups


if __name__ == '__main__':
    np.random.seed(777)
    train_dataset, test_dataset, user_groups = get_dataset(num_data=40000, num_users=100, iid=False, num_share=4000,
                                                           l=2, unequal=False, share_l=2)
