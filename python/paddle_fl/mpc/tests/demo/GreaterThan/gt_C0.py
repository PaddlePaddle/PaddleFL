import unittest
from multiprocessing import Manager
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

class TestOpGreaterThan_C0(test_op_base.TestOpBase):

    def sub_gt_C0(self, **kwargs):

        role = 0
        num = 10
        d_1 = np.load('data_C0_P0.npy',allow_pickle=True)
        d_2 = np.load('data_C0_P1.npy',allow_pickle=True)
        d_zero = np.full((num), fill_value=0).astype('float32')
        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[num], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[num], dtype='int64')
        zero = fluid.data(name='zero', shape=[num], dtype='float32')
        #op_sub = x - y
        op_sub = pfl_mpc.layers.elementwise_sub(x=x, y=y)
        exe = fluid.Executor(place=fluid.CPUPlace())
        d_tmp = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_sub])
        # op_gt = tmp > 0
        op_gt = pfl_mpc.layers.greater_than(x=x, y=zero)
        results = exe.run(feed={'x': d_tmp[0],'y':d_2, 'zero': d_zero}, fetch_list=[op_gt])
        f = open('out.txt', 'w')
        for i in range(num):
            f.write(str(results[0][i]) + '\n')
        f.close()

    def test_sub_gt_C0(self):
        ret = self.multi_party_run0(target=self.sub_gt_C0)

if __name__ == '__main__':
    unittest.main()