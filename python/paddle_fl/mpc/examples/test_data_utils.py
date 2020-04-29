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

# export
import sys

import data_utils as du
import numpy as np

plaintext = 2.9
print("Plaintext: {0}".format(plaintext))

shares = du.share(plaintext)
print("Shares: {0}".format(shares))

revealed_text = du.reveal(shares)
print("Revealed plaintext: {0}".format(revealed_text))
