import unittest
from multiprocessing import Manager
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

class TestOpAdd2_C0(test_op_base.TestOpBase):

    def add2_C0(self, **kwargs):

        role = 0
        num = 100
        d_1 = np.load('data_C0_P0.npy',allow_pickle=True)
        d_2 = np.load('data_C0_P1.npy',allow_pickle=True)
        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[num], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[num], dtype='int64')
        # math_add = x + y
        math_add = pfl_mpc.layers.elementwise_add(x, y, axis=1)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[math_add])
        np.save('result_C0.npy', results[0])

    def test_add2_C0(self):
        ret = self.multi_party_run0(target=self.add2_C0)

if __name__ == '__main__':
    unittest.main()