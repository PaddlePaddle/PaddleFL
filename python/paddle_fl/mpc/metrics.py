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
MPC Metrics
"""

import paddle.fluid.metrics
from paddle.fluid.metrics import MetricBase

import numpy as np
import scipy


__all__ = [
    'KSstatistic',
    'Auc',
]


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


class KSstatistic(MetricBase):
    """
    The KSstatistic is for binary classification.
    Refer to https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov%E2%80%93Smirnov_statistic
    Please notice that the KS statistic is implemented with scipy.

    The `KSstatistic` function creates 2 local variables, `data1`, `data2`
    which is predictions of positive samples and negative samples respectively
    that are used to compute the KS statistic.

    Args:
        name (str, optional): Metric name. For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle_fl.mpc
            import numpy as np

            # suppose that batch_size is 128
            batch_num = 100
            batch_size = 128

            for batch_id in range(batch_num):

                class0_preds = np.random.random(size = (batch_size, 1))
                class1_preds = 1 - class0_preds

                preds = np.concatenate((class0_preds, class1_preds), axis=1)

                labels = np.random.randint(2, size = (batch_size, 1))

                # init the KSstatistic for each batch
                # to get global ks statistic, init ks before for-loop
                ks = paddle_fl.mpc.metrics.KSstatistic('ks')
                ks.update(preds = preds, labels = labels)

                # shall be some score closing to 0.1 as the preds are randomly assigned
                print("ks statistic for iteration %d is %.2f" % (batch_id, ks.eval()))

    """

    def __init__(self, name=None):
        super(KSstatistic, self).__init__(name=name)
        self._data1 = []
        self._data2 = []

    def update(self, preds, labels):
        """
        Update the auc curve with the given predictions and labels.

        Args:
             preds (numpy.array): an numpy array in the shape of
             (batch_size, 2), preds[i][j] denotes the probability of
             classifying the instance i into the class j.

             labels (numpy.array): an numpy array in the shape of
             (batch_size, 1), labels[i] is either o or 1, representing
             the label of the instance i.
        """
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        if not _is_numpy_(preds):
            raise ValueError("The 'predictions' must be a numpy ndarray.")

        data1 = [preds[i, 1] for i, lbl in enumerate(labels) if lbl]
        data2 = [preds[i, 1] for i, lbl in enumerate(labels) if not lbl]

        self._data1 += data1
        self._data2 += data2

    def eval(self):
        """
        Return the area (a float score) under auc curve

        Return:
            float: the area under auc curve
        """

        return scipy.stats.ks_2samp(self._data1, self._data2).statistic


Auc = paddle.fluid.metrics.Auc
