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
Evaluate accuracy.
"""
import numpy as np
import logging

import paddle_fl.mpc as pfl_mpc

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def evaluate_auc(file1, file2):
    """
    evaluate accuracy
    """
    auc_metric = pfl_mpc.metrics.Auc("ROC")
    label = np.loadtxt(file1, delimiter='\n')
    class1_preds = np.loadtxt(file2, delimiter='\n')
    label_data = label.reshape(label.shape[0], 1)
    class1_preds = class1_preds.reshape(label_data.shape)

    class0_preds = 1 - class1_preds
    preds = np.concatenate((class0_preds, class1_preds), axis=1)

    auc_metric.update(preds=preds, labels=label_data)
    logger.info("Evaluate AUC: ")
    auc_value = auc_metric.eval()
    logger.info(auc_value)
    return auc_value


def evaluate_accuracy(file1, file2):
    """
    evaluate accuracy
    """
    count = 0
    same_count = 0
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    while 1:
        line1 = f1.readline().strip('\n')
        line2 = f2.readline().strip('\n')
        if (not line1) or (not line2):
            break
        count += 1
        if int(float(line1)) == int(1 if float(line2) > 0.5 else 0):
            same_count += 1
    logger.info("evaluate accuracy: ")
    logger.info(float(same_count) / count)
    return float(same_count) / count


if __name__ == '__main__':
    evaluate_accuracy("./mpc_infer_data/label_criteo",
                      "./mpc_infer_data/label_mpc")
    evaluate_auc("./mpc_infer_data/label_criteo", "./mpc_infer_data/label_mpc")
