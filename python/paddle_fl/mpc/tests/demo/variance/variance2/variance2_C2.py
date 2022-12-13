import unittest
from multiprocessing import Manager
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

class TestOpVariance2_C2(test_op_base.TestOpBase):
    #avg = (num0 * avg0 + num1 * avg1) / (num0 + num1)
    #prod = ((num0 - 1) * variance0 + (num1 - 1) * variance1 + num0 * (avg0 - avg) * (avg0 - avg) + num1 * (avg1 - avg) * (avg1 - avg)) / (num0 + num1 - 1)

    def variance2_C2(self, **kwargs):

        role = 2
        d_1 = np.load('data_C2_P0_avg.npy',allow_pickle=True) #avg0
        d_2 = np.load('data_C2_P0_variance.npy',allow_pickle=True) #variance0
        d_3 = np.load('data_C2_P1_avg.npy',allow_pickle=True) #avg1
        d_4 = np.load('data_C2_P1_variance.npy',allow_pickle=True) #variance1
        d_tmp = np.load('data_C2_tmp.npy',allow_pickle=True) #tmp
        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[1], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[1], dtype='int64')
        # math_add = x + y
        math_add = pfl_mpc.layers.elementwise_add(x, y, axis=1)
        math_sub = pfl_mpc.layers.elementwise_sub(x, y, axis=1)
        math_mul = pfl_mpc.layers.mul(x,y)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results_0 = exe.run(feed={'x': d_1, 'y': d_tmp[:,1].reshape(2,1)}, fetch_list=[math_mul])# (avg0 * num0)
        results_1 = exe.run(feed={'x': d_3, 'y': d_tmp[:,2].reshape(2,1)}, fetch_list=[math_mul])# (avg1 * num1)
        results = exe.run(feed={'x': results_0[0], 'y': results_1[0]}, fetch_list=[math_add]) # (avg0 * num0 + avg1 * num1)
        avg = exe.run(feed={'x': results[0], 'y': d_tmp[:,0].reshape(2,1)}, fetch_list=[math_mul])# (avg0 * num0 + avg1 * num1) / (num0 + num1)
        d_avg_0 = exe.run(feed={'x': d_1, 'y': avg[0]}, fetch_list=[math_sub]) # avg0 - avg
        d_tmp_1 = exe.run(feed={'x': d_avg_0[0], 'y': d_avg_0[0]}, fetch_list=[math_mul]) # (avg0 - avg)*(avg0 - avg)
        d_tmp_2 = exe.run(feed={'x': d_tmp[:,1].reshape(2,1), 'y': d_tmp_1[0]}, fetch_list=[math_mul]) # num0 * (avg0 - avg) * (avg0 - avg)
        d_avg_1 = exe.run(feed={'x': d_3, 'y': avg[0]}, fetch_list=[math_sub]) # avg1 - avg
        d_tmp_3 = exe.run(feed={'x': d_avg_1[0], 'y': d_avg_1[0]}, fetch_list=[math_mul]) # (avg1 - avg)*(avg1 - avg)
        d_tmp_4 = exe.run(feed={'x': d_tmp[:,2].reshape(2,1), 'y': d_tmp_3[0]}, fetch_list=[math_mul]) # num1 * (avg1 - avg) * (avg1 - avg)
        d_tmp_5 = exe.run(feed={'x': d_tmp_2[0], 'y': d_tmp_4[0]}, fetch_list=[math_add])  # num0 * (avg0 - avg) * (avg0 - avg) + num1 * (avg1 - avg) * (avg1 - avg)
        d_tmp_6 = exe.run(feed={'x': d_tmp[:,3].reshape(2,1), 'y': d_2}, fetch_list=[math_mul]) # (num0 - 1) * variance0
        d_tmp_7 = exe.run(feed={'x': d_tmp[:,4].reshape(2,1), 'y': d_4}, fetch_list=[math_mul]) # (num1 - 1) * variance1
        d_tmp_8 = exe.run(feed={'x': d_tmp_6[0], 'y': d_tmp_7[0]}, fetch_list=[math_add])  # (num0 - 1) * variance0 + (num1 - 1) * variance1
        d_tmp_9 = exe.run(feed={'x': d_tmp_5[0], 'y': d_tmp_8[0]}, fetch_list=[math_add])  # d_tmp_5 + d_tmp_8
        results = exe.run(feed={'x': d_tmp_9[0], 'y': d_tmp[:,5].reshape(2,1)}, fetch_list=[math_mul])
        np.save('result_C2.npy', results[0])

    def test_variance2_C2(self):
        ret = self.multi_party_run0(target=self.variance2_C2)

if __name__ == '__main__':
    unittest.main()