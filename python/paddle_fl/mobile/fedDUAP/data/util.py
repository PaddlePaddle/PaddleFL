import math
import numpy as np

def show_data(train_dataset, user_groups, num_users=20):
    """
    show the first num_users deivce data distribution and the last one
    :param train_dataset:
    :return:
    """
    client_noniid_degree, share_noniid_degree = get_client_server_noniid_degree(train_dataset, user_groups)
    print("noniid degree share / client : {:.4f} / {:.4f}".format(share_noniid_degree, client_noniid_degree))

    num_users = min(len(user_groups) - 1, num_users)

    # count devices
    user_cnt = {}
    for i in range(num_users):
        user_cnt[i] = [0] * len(train_dataset.classes)
        for idx in user_groups[i]:
            _, label = train_dataset[int(idx)]
            user_cnt[i][int(label)] += 1

    # 统计共享数据
    i = len(user_groups) - 1
    user_cnt[i] = [0] * len(train_dataset.classes)
    for idx in user_groups[i]:
        _, label = train_dataset[int(idx)]
        user_cnt[i][int(label)] += 1

    # 打印显示
    idx = 0
    for k, v in user_cnt.items():
        print(f"user {k}:", end=" ")
        for i in range(len(v)):
            if v[i] != 0:
                print(f"{i}:{v[i]}", end=" ")
        print(" | ", end=" ")
        idx += 1
        if idx % 5 == 0:
            print()
    print()

def get_client_server_noniid_degree(train_dataset, user_groups):
    """
    获取给定分配，客户端和服务端的 noniid 情况
    :param train_dataset:
    :param user_groups:
    :param num_users:
    :return:
    """
    total_idxs = np.array([])
    for user, idxs in user_groups.items():
        if user == len(user_groups) - 1:
            continue
        total_idxs = np.append(total_idxs, idxs)

    global_distribution = get_distribution(train_dataset, total_idxs)
    share_distribution = get_distribution(train_dataset, user_groups[len(user_groups) - 1])
    client_distribution = get_distribution(train_dataset, user_groups[0])
    share_noniid_degree = get_noniid_degree(share_distribution, global_distribution)
    client_noniid_degree = get_noniid_degree(client_distribution, global_distribution)
    return client_noniid_degree, share_noniid_degree


def get_distribution(train_dataset, idxs):
    cnt_category = [0] * len(train_dataset.classes)
    for idx in idxs:
        _, label = train_dataset[int(idx)]
        cnt_category[int(label)] += 1
    total = sum(cnt_category)
    if total == 0:
        return cnt_category
    distribution = [round(cnt / total, 3) for cnt in cnt_category]
    return distribution

def get_noniid_degree(d1, d2):
    # sum = 0
    # for i in range(len(d1)):
    #     sum += (d1[i] - d2[i]) * (d1[i] - d2[i])
    return JS_divergence(d1, d2)

def get_global_distribution(train_dataset, user_groups):
    """
    获取总体客户端数据的分布
    :param train_dataset:
    :param user_groups:
    :return:
    """
    total_idxs = np.array([])
    for user, idxs in user_groups.items():
        if user == len(user_groups) - 1:
            continue
        total_idxs = np.append(total_idxs, idxs)

    global_distribution = get_distribution(train_dataset, total_idxs)
    return global_distribution

def KL_divergence(p1, p2):
    d = 0
    for i in range(len(p1)):
        if p2[i] == 0 or p1[i] == 0:
            continue
        d += p1[i] * math.log(p1[i]/p2[i], 2)
    return d

def JS_divergence(p1, p2):
    p3 = []
    for i in range(len(p1)):
        p3.append((p1[i] + p2[i])/2)
    return KL_divergence(p1, p3)/2 + KL_divergence(p2, p3)/2

def get_target_users_distribution(train_dataset, user_groups, user_ids):
    """
    获取指定客户端总体数据的分布
    :param train_dataset:
    :param user_groups:
    :return:
    """
    total_idxs = np.array([])
    for user_id in user_ids:
        total_idxs = np.append(total_idxs, user_groups[user_id])

    users_distribution = get_distribution(train_dataset, total_idxs)
    return users_distribution

if __name__=="__main__":
    from data.cifar100.cifar100_data import get_dataset
    train_dataset, test_dataset, user_groups = get_dataset(num_data=40000, num_users=100, iid=False, num_share=4000,
                                                           l=2, unequal=False, share_l=10)
    global_distribution = get_global_distribution(train_dataset, user_groups)
    users_distribution = get_target_users_distribution(train_dataset, user_groups, range(0, 10))
    server_distribution = get_target_users_distribution(train_dataset, user_groups, [100])
    shared_data_noniid_degree = get_noniid_degree(global_distribution, server_distribution)
    users_noniid_degree = get_noniid_degree(global_distribution, users_distribution)
    print(users_noniid_degree)