import unittest
from multiprocessing import Manager
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

class TestOpMul2_C1(test_op_base.TestOpBase):

    def mul2_C1(self, **kwargs):

        role = 1
        num = 100
        d_1 = np.load('data_C1_P0.npy',allow_pickle=True)
        d_2 = np.load('data_C1_P1.npy',allow_pickle=True)

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[num], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[num], dtype='int64')
        # math_mul = x * y
        math_mul = pfl_mpc.layers.elementwise_mul(x, y)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[math_mul])
        np.save('result_C1.npy', results[0])

    def test_mul2_C1(self):
        ret = self.multi_party_run1(target=self.mul2_C1)

if __name__ == '__main__':
    unittest.main()