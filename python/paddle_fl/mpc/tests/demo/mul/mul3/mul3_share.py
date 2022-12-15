import numpy as np
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

if __name__ == '__main__':
        with open(r'Input-P0.list', 'r') as file:
            content_list_0 = file.readlines()
        data_1 = np.array([float(x) for x in content_list_0])

        data_1_shares = aby3.make_shares(data_1)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        np.save('data_C0_P0.npy', data_1_all3shares[0])
        np.save('data_C1_P0.npy', data_1_all3shares[1])
        np.save('data_C2_P0.npy', data_1_all3shares[2])

        with open(r'Input-P1.list', 'r') as file:
            content_list_1 = file.readlines()
        data_2 = np.array([float(x) for x in content_list_1])

        data_2_shares = aby3.make_shares(data_2)
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])
        np.save('data_C0_P1.npy', data_2_all3shares[0])
        np.save('data_C1_P1.npy', data_2_all3shares[1])
        np.save('data_C2_P1.npy', data_2_all3shares[2])

        with open(r'Input-P2.list', 'r') as file:
            content_list_2 = file.readlines()
        data_3 = np.array([float(x) for x in content_list_2])

        data_3_shares = aby3.make_shares(data_3)
        data_3_all3shares = np.array([aby3.get_shares(data_3_shares, i) for i in range(3)])
        np.save('data_C0_P2.npy', data_3_all3shares[0])
        np.save('data_C1_P2.npy', data_3_all3shares[1])
        np.save('data_C2_P2.npy', data_3_all3shares[2])

        contentall = data_1 * data_2 * data_3
        print(contentall)