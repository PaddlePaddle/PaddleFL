# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
mpc metric op layers.
"""
from paddle.fluid.data_feeder import check_type, check_dtype
from paddle.fluid.initializer import Constant
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper

__all__ = ['precision_recall']

def precision_recall(input, label, threshold=0.5):
    """
    Precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances.
    Recall (also known as sensitivity) is the fraction of
    relevant instances that have been retrieved over the
    total amount of relevant instances
    F1-score is a measure of a test's accuracy.
    It is calculated from the precision and recall of the test.
    Refer to:
    https://en.wikipedia.org/wiki/Precision_and_recall
    https://en.wikipedia.org/wiki/F1_score

    Noted that this class manages the metrics only for binary classification task.
    Noted that in both precision and recall, define 0/0 equals to 0.

    Args:
        input (Variable): ciphtext predicts for 1 in binary classification.
        label (Variable): labels in ciphertext.
        threshold (float): predict threshold.
    Returns:
        batch_out (Variable): plaintext of batch metrics [precision, recall, f1-score]
            Note that values in batch_out are fixed-point number.
            To get float type values, div fetched batch_out by
            3 * mpc_data_utils.mpc_one_share (which equals to 2**16).
        acc_out (Variable): plaintext of accumulated metrics [precision, recall, f1-score]
            To get float type values, div fetched acc_out by
            3 * mpc_data_utils.mpc_one_share (which equals to 2**16).

    Examples:
        .. code-block:: python
            import sys
            import numpy as np
            import paddle.fluid as fluid
            import paddle_fl.mpc as pfl_mpc
            import mpc_data_utils as mdu

            role = int(sys.argv[1])

            redis_server = "127.0.0.1"
            redis_port = 9937
            loop = 5
            np.random.seed(0)

            input_size = [100]

            threshold = 0.6

            preds, labels = [], []
            preds_cipher, labels_cipher = [], []
            #simulating mpc share

            share = lambda x: np.array([x * mdu.mpc_one_share] * 2).astype('int64').reshape([2] + input_size)
            for _ in range(loop):

                preds.append(np.random.random(input_size))
                labels.append(np.rint(np.random.random(input_size)))
                preds_cipher.append(share(preds[-1]))
                labels_cipher.append(share(labels[-1]))

            pfl_mpc.init("aby3", role, "localhost", redis_server, redis_port)
            x = pfl_mpc.data(name='x', shape=input_size, dtype='int64')
            y = pfl_mpc.data(name='y', shape=input_size, dtype='int64')
            out0, out1 = pfl_mpc.layers.precision_recall(input=x, label=y, threshold=threshold)
            exe = fluid.Executor(place=fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            for i in range(loop):
                batch_res, acc_res = exe.run(feed={'x': preds_cipher[i], 'y': labels_cipher[i]},
                        fetch_list=[out0, out1])
                fixed_point_one = 3.0 * mdu.mpc_one_share
                # result could be varified by calcuatling metrics with plaintext preds, labels
                print(batch_res / fixed_point_one , acc_res / fixed_point_one)

    """
    helper = MpcLayerHelper("precision_recall", **locals())

    dtype = helper.input_dtype()

    check_dtype(dtype, 'input', ['int64'], 'precision_recall')
    check_dtype(dtype, 'label', ['int64'], 'precision_recall')

    batch_out = helper.create_mpc_variable_for_type_inference(dtype=input.dtype)
    acc_out = helper.create_mpc_variable_for_type_inference(dtype=input.dtype)

    stat = helper.create_global_mpc_variable(
            persistable=True,
            dtype='int64', shape=[3],
            )

    helper.set_variable_initializer(stat, Constant(value=0))

    op_type = 'precision_recall'

    helper.append_op(
        type='mpc_' + op_type,
        inputs={
            "Predicts": input,
            "Labels": label,
            "StatesInfo": stat,
            },
        outputs={
            "BatchMetrics": batch_out,
            "AccumMetrics": acc_out,
            "AccumStatesInfo": stat,
             },
        attrs={
            "threshold": threshold,
            "class_number": 1,
        })

    return batch_out, acc_out
