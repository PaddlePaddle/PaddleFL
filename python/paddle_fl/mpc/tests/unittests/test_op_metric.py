#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module test metric op.

"""
import unittest
import numpy as np
import test_op_base
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import mpc_data_utils as mdu
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


def precision_recall_naive(input, label, threshold=0.5, stat=None):
    pred = input - (threshold - 0.5)
    pred = np.maximum(0, pred)
    pred = np.minimum(1, pred)
    idx = np.rint(pred)
    tp = np.sum(idx * label)
    fp = np.sum(idx) - tp
    fn = np.sum(label) - tp

    def calc_precision(tp, fp):
        return tp / (tp + fp) if tp + fp > 0 else 0.0

    def calc_recall(tp, fn):
        return tp / (tp + fn) if tp + fn > 0 else 0.0

    def calc_f1(precision, recall):
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    p_batch, r_batch = calc_precision(tp, fp), calc_recall(tp, fn)
    f_batch = calc_f1(p_batch, r_batch)

    p_acc, r_acc, f_acc = p_batch, r_batch, f_batch

    if stat:
        tp += stat[0]
        fp += stat[1]
        fn += stat[2]

        p_acc, r_acc = calc_precision(tp, fp), calc_recall(tp, fn)
        f_acc = calc_f1(p_acc, r_acc)

    new_stat = [tp, fp, fn]

    return np.array([p_batch, r_batch, f_batch, p_acc, r_acc, f_acc]), new_stat


class TestOpPrecisionRecall(test_op_base.TestOpBase):

    def precision_recall(self, **kwargs):
        """
        precision_recall op ut
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        preds = kwargs['preds']
        labels = kwargs['labels']
        loop = kwargs['loop']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=self.input_size, dtype='int64')
        y = pfl_mpc.data(name='y', shape=self.input_size, dtype='int64')
        out0, out1 = pfl_mpc.layers.precision_recall(input=x, label=y, threshold=self.threshold)
        exe = fluid.Executor(place=fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        for i in range(loop):
            batch_res, acc_res = exe.run(feed={'x': preds[i], 'y': labels[i]},
                    fetch_list=[out0, out1])

        self.assertTrue(np.allclose(batch_res * (2 ** -16), self.exp_res[0][:3], atol=1e-4))
        self.assertTrue(np.allclose(acc_res* (2 ** -16), self.exp_res[0][3:], atol=1e-4))

    def n_batch_test(self, n):
        self.input_size = [100]

        self.threshold = np.random.random()
        preds, labels = [], []
        self.exp_res = (0, [0] * 3)
        share = lambda x: np.array([x * aby3.MPC_ONE_SHARE] * 2).astype('int64').reshape(
                [2] + self.input_size)

        for _ in range(n):
            preds.append(np.random.random(self.input_size))
            labels.append(np.rint(np.random.random(self.input_size)))
            self.exp_res = precision_recall_naive(preds[-1], labels[-1],
                    self.threshold, self.exp_res[-1])
            preds[-1] = share(preds[-1])
            labels[-1] = share(labels[-1])

        ret = self.multi_party_run(target=self.precision_recall,
                preds=preds, labels=labels, loop=n)

        self.assertEqual(ret[0], True)

    def test_1(self):
        self.n_batch_test(1)

    def test_2(self):
        self.n_batch_test(2)


if __name__ == '__main__':
    unittest.main()

