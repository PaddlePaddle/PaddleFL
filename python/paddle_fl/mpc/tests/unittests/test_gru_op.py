#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import math
import functools
from op_test import OpTest
from multiprocessing import Manager
import test_op_base
import paddle.fluid as fluid
import paddle.fluid.core as core


SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


def identity(x):
    return x


def sigmoid(x):
    y = np.copy(x)
    y[x < SIGMOID_THRESHOLD_MIN] = SIGMOID_THRESHOLD_MIN
    y[x > SIGMOID_THRESHOLD_MAX] = SIGMOID_THRESHOLD_MAX
    return 1. / (1. + np.exp(-y))


def tanh(x):
    y = -2. * x
    y[y > EXP_MAX_INPUT] = EXP_MAX_INPUT
    return (2. / (1. + np.exp(y))) - 1.


def relu(x):
    return np.maximum(x, 0)


ACTIVATION = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}

def gru(
        input,  # T x 3D
        lod,  # 1 x N
        h0,  # N x D
        weight,  # D x 3D
        bias,  # 1 x 3D
        is_reverse,
        act_state,
        act_gate,
        dtype='float32',
        origin_mode=False):
    def _seq_to_batch(lod, is_reverse):
        idx_in_seq_list = []
        seq_lens = lod[0]
        seq_starts = [0]
        for i in range(len(seq_lens)):
            seq_starts.append(seq_starts[-1] + seq_lens[i])
        sorted_seqs = sorted(
            list(range(len(seq_lens))),
            key=functools.cmp_to_key(lambda x, y: seq_lens[y] - seq_lens[x]))
        num_batch = seq_lens[sorted_seqs[0]]
        for batch_idx in range(num_batch):
            idx_in_seq = []
            for i in range(len(seq_lens)):
                if seq_lens[sorted_seqs[i]] <= batch_idx:
                    break
                idx = (seq_starts[sorted_seqs[i] + 1] - 1 - batch_idx
                       ) if is_reverse else (
                           seq_starts[sorted_seqs[i]] + batch_idx)
                idx_in_seq.append(idx)
            idx_in_seq_list.append(idx_in_seq)
        return idx_in_seq_list, sorted_seqs

    def _step(x, h_p, w, b, act_state, act_gate):
        T = x.shape[0]
        D = w.shape[0]
        g = x + np.tile(b, (T, 1))
        w_u_r = w.flatten()[:D * D * 2].reshape((D, D * 2))
        u_r = act_gate(np.dot(h_p, w_u_r) + g[:, :D * 2])
        u = u_r[:, :D]
        r = u_r[:, D:D * 2]
        r_h_p = r * h_p
        w_c = w.flatten()[D * D * 2:].reshape((D, D))
        c = act_state(np.dot(r_h_p, w_c) + g[:, D * 2:])
        g = np.hstack((u_r, c))
        if origin_mode:
            h = (1 - u) * c + u * h_p
        else:
            h = u * c + (1 - u) * h_p
        return g, r_h_p, h

    T = sum(lod[0])
    N = len(lod[0])
    D = weight.shape[0]
    batch_gate = np.zeros((T, 3 * D), dtype=dtype)
    batch_reset_hidden_prev = np.zeros((T, D), dtype=dtype)
    batch_hidden = np.zeros((T, D), dtype=dtype)
    hidden = np.zeros((T, D), dtype=dtype)

    idx_in_seq_list, sorted_seqs = _seq_to_batch(lod, is_reverse)
    h_p = h0[[seq for seq in sorted_seqs if lod[0][seq] > 0]]

    max_seq_len = len(idx_in_seq_list)
    end_idx = 0
    for batch_idx in range(max_seq_len):
        x = input[idx_in_seq_list[batch_idx]]
        g, r_h_p, h = _step(x, h_p, weight, bias, act_state, act_gate)
        if batch_idx < (max_seq_len - 1):
            h_p = h[:len(idx_in_seq_list[batch_idx + 1])]
        start_idx = end_idx
        end_idx = start_idx + len(idx_in_seq_list[batch_idx])
        batch_gate[start_idx:end_idx] = g
        batch_reset_hidden_prev[start_idx:end_idx] = r_h_p
        batch_hidden[start_idx:end_idx] = h
        hidden[idx_in_seq_list[batch_idx]] = h
    return batch_gate, batch_reset_hidden_prev, batch_hidden, hidden


class TestGRUOp(OpTest):
    def set_confs(self):
        pass

    def setUp(self):
        OpTest.setUp(self)
        self.op_type = "mpc_gru"
        self.lod = [[1, 2, 1]]

        self.D = 2
        self.is_reverse = False
        self.with_h0 = False
        self.with_bias = False
        self.act_state = 'relu'
        self.act_gate = 'sigmoid'
        self.dtype = 'float32'
        self.origin_mode = False
        self.set_confs()

        T = sum(self.lod[0]) # 7
        N = len(self.lod[0]) # 3
        input = np.random.rand(T, 3 * self.D).astype(self.dtype)
        weight = np.random.rand(self.D, 3 * self.D).astype(self.dtype)

        bias = np.random.rand(
            1, 3 * self.D).astype(self.dtype) if self.with_bias else np.zeros(
                (1, 3 * self.D), dtype=self.dtype)
        h0 = np.random.rand(
            N, self.D).astype(self.dtype) if self.with_h0 else np.zeros(
                (N, self.D), dtype=self.dtype)

        self.mpc_input = self.lazy_share(input) # shape 2 * 7 * 6
        # transpose mpc tensor to set lod for input
        mpc_input_trans = np.transpose(self.mpc_input, [1, 0, 2]) # shape 7 * 2 * 6
        self.mpc_weight = self.lazy_share(weight)
        mpc_bias = self.lazy_share(bias)
        mpc_h0 = self.lazy_share(h0)

        batch_gate, batch_reset_hidden_prev, batch_hidden, hidden = gru(
            input, self.lod, h0, weight, bias, self.is_reverse,
            ACTIVATION[self.act_state], ACTIVATION[self.act_gate], self.dtype,
            self.origin_mode)

        self.inputs = {'Input': (mpc_input_trans, self.lod), 'Weight': self.mpc_weight}

        if self.with_bias:
            self.inputs['Bias'] = mpc_bias

        if self.with_h0:
            self.inputs['H0'] = mpc_h0

        self.outputs = {
            'Hidden': (hidden, self.lod),
            'BatchGate': batch_gate,
            'BatchResetHiddenPrev': batch_reset_hidden_prev,
            'BatchHidden': batch_hidden,
        }

        self.attrs = {
            'activation': self.act_state,
            'gate_activation': 'sigmoid',
            'is_reverse': self.is_reverse,
            'origin_mode': self.origin_mode
        }

    def _check_output(self):
        place = fluid.CPUPlace()
        self.check_output_with_place(place, atol=1e-1)

    def test_check_grad(self):
        # set output type to 'int64'
        # TODO: if not set outputs type to 'int64', exception will throw
        self.outputs = {
            'Hidden': np.array([1]).astype('int64'),
            'BatchGate': np.array([1]).astype('int64'),
            'BatchResetHiddenPrev': np.array([1]).astype('int64'),
            'BatchHidden': np.array([1]).astype('int64'),
        }
        place = core.CPUPlace()
        # transpose_input_list specifies transposed mpc LodTensor
        self.check_grad_with_place(
            place, ['Input', 'Weight'], ['Hidden'],
            max_relative_error=50, check_dygraph=False, numeric_grad_delta=0.005,
            transpose_input_list=["Input"])


class TestGRUOriginMode(TestGRUOp):
    def set_confs(self):
        self.origin_mode = True


class TestGRUOp2Len0(TestGRUOp):
    def set_confs(self):
        self.lod = [[1, 0, 1]]


class TestGRUOp2OriginModeLen0(TestGRUOp):
    def set_confs(self):
        self.lod = [[0, 2, 1]]
        self.origin_mode = True


class TestGRUOp2OriginModeLastLen0(TestGRUOp):
    def set_confs(self):
        self.lod = [[0, 2, 0]]
        self.origin_mode = True


class TestGRUOpWithInitial(TestGRUOp):
    def set_confs(self):
        self.with_h0 = True

    def test_check_grad(self):
        # set output type to 'int64'
        # TODO: if not set outputs type to 'int64', exception will throw
        self.outputs = {
            'Hidden': np.array([1]).astype('int64'),
            'BatchGate': np.array([1]).astype('int64'),
            'BatchResetHiddenPrev': np.array([1]).astype('int64'),
            'BatchHidden': np.array([1]).astype('int64'),
        }
        place = core.CPUPlace()
        self.check_grad_with_place(
            place, ['Input', 'Weight', 'H0'], ['Hidden'],
            max_relative_error=50, check_dygraph=False,
            transpose_input_list=["Input"])


class TestGRUOpWithBias(TestGRUOp):
    def set_confs(self):
        self.with_bias = True

    def test_check_grad(self):
        # set output type to 'int64'
        # TODO: if not set outputs type to 'int64', exception will throw
        self.outputs = {
            'Hidden': np.array([1]).astype('int64'),
            'BatchGate': np.array([1]).astype('int64'),
            'BatchResetHiddenPrev': np.array([1]).astype('int64'),
            'BatchHidden': np.array([1]).astype('int64'),
        }
        place = core.CPUPlace()
        self.check_grad_with_place(
            place, ['Input', 'Weight', 'Bias'], ['Hidden'],
            max_relative_error=50, check_dygraph=False,
            transpose_input_list=["Input"])
"""
class TestGRUOpReverse(TestGRUOp):
    def set_confs(self):
        self.is_reverse = True


class TestGRUOpReverseOriginMode(TestGRUOp):
    def set_confs(self):
        self.is_reverse = True
        self.origin_mode = True
"""

if __name__ == "__main__":
    unittest.main()
