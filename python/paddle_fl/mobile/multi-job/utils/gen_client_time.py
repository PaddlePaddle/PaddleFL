import numpy as np
import json


def gen_time0(MAC, Index):
    freq = [1 / 0.5, 1 / 1, 1 / 1.5, 1 / 2]  # Heterogeneous device types 0.5, 1, 1.5, 2
    tau = 5  # number of local updates
    D = [30, 64, 20]  # mini-batch size # 30s, 70s, 20s

    tcp_lst = []
    for v in freq:
        for i, value in enumerate(D):
            a = MAC[i] * v  # ms/sample
            mu = 1 / a

            d = value
            lam = mu / (tau * d)
            beta = 1 / lam
            if len(tcp_lst) < len(MAC):
                tcp_lst.append(list(np.random.exponential(beta, size=int(100 / len(freq))) / 10 + 10))
            else:
                tcp_lst[i].extend(list(np.random.exponential(beta, size=int(100 / len(freq))) / 10 + 10))

    ord = np.argsort(tcp_lst[0])  # the first job, the return increasing index
    for i, v in enumerate(tcp_lst[1:]):  # Other jobs are sorted in increasing order of time
        tcp_lst[i + 1] = np.sort(v)

    client_time = [[] for i in range(100)]
    for i, index in enumerate(ord):
        client_time[index] = [tcp_lst[0][index], tcp_lst[1][i], tcp_lst[2][i]]

    print(len(client_time), client_time)
    with open("client_time_" + Index + ".json", "w") as f:
        json.dump(client_time, f)


if __name__ == "__main__":
    MAC_A = [3.0, 5.0, 2.0]
    MAC_B = [3.2, 5.9, 2.2]
    gen_time0(MAC_A, "A")
    gen_time0(MAC_B, "B")
