import numpy as np
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')


if __name__ == '__main__':
        return_results = list()
        result_C0 = np.load('result_C0.npy', allow_pickle=True)
        result_C1 = np.load('result_C1.npy',allow_pickle=True)
        result_C2 = np.load('result_C2.npy',allow_pickle=True)
        return_results.append(result_C0)
        return_results.append(result_C1)
        return_results.append(result_C2)
        revealed = aby3.reconstruct(np.array(return_results))
        print(revealed)