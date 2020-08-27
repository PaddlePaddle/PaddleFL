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
Decrypt Prediction Data.
"""
import sys
import os
from process_data import decrypt_data_to_file

decrypt_file=sys.argv[1]
BATCH_SIZE=128
class_num = 2

mpc_infer_data_dir = "./mpc_infer_data/"
prediction_file = mpc_infer_data_dir + "mnist_debug_prediction"

if os.path.exists(decrypt_file):
    os.remove(decrypt_file)

if class_num == 2:
    decrypt_data_to_file(prediction_file, (BATCH_SIZE,), decrypt_file)
elif class_num == 10:
    decrypt_data_to_file(prediction_file, (BATCH_SIZE, 10), decrypt_file)
else:
    raise ValueError("class_num should be 2 or 10, but received {}.".format(class_num))

