import unittest
from multiprocessing import Manager
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

class TestOpMax3_C0(test_op_base.TestOpBase):

    def max3_C0(self, **kwargs):

        role = 0
        d_1 = np.load('data_C0_P0.npy',allow_pickle=True)
        d_2 = np.load('data_C0_P1.npy',allow_pickle=True)
        d_3 = np.load('data_C0_P2.npy', allow_pickle=True)
        d_zero = np.full((1), fill_value=0).astype('float32')
        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[1], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[1], dtype='int64')
        zero = fluid.data(name='zero', shape=[1], dtype='float32')
        #op_sub = x - y
        op_sub = pfl_mpc.layers.elementwise_sub(x=x, y=y)
        exe = fluid.Executor(place=fluid.CPUPlace())
        d_tmp0 = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_sub])
        # op_gt = tmp > 0
        op_gt = pfl_mpc.layers.greater_than(x=x, y=zero)
        results0 = exe.run(feed={'x': d_tmp0[0],'y':d_2, 'zero': d_zero}, fetch_list=[op_gt])
        if results0[0] == 1:
            d_tmp1 = exe.run(feed={'x': d_3, 'y': d_1, 'zero': d_zero}, fetch_list=[op_sub])
            results1 = exe.run(feed={'x': d_tmp1[0], 'y': d_2, 'zero': d_zero}, fetch_list=[op_gt])
        else:
            d_tmp1 = exe.run(feed={'x': d_3, 'y': d_2, 'zero': d_zero}, fetch_list=[op_sub])
            results1 = exe.run(feed={'x': d_tmp1[0], 'y': d_2, 'zero': d_zero}, fetch_list=[op_gt])
        if results1[0] == 1:
            np.save('result_C0.npy', d_3)
        else:
            if results0[0] == 1:
                np.save('result_C0.npy', d_1)
            else:
                np.save('result_C0.npy', d_2)

    def test_max3_C0(self):
        ret = self.multi_party_run0(target=self.max3_C0)

if __name__ == '__main__':
    unittest.main()