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

import numpy as np
from paddle_serving_client import Client

client = Client()
client.load_client_config("imdb_client_conf/serving_client_conf.prototxt")
client.connect(["127.0.0.1:9292"])

data_dict = {}

for i in range(3):
    data_dict[str(i)] = np.random.rand(1, 5).astype('float32')

fetch_map = client.predict(
    feed={"0": data_dict['0'],
          "1": data_dict['1'],
          "2": data_dict['2']},
    fetch=["fc_2.tmp_2"])

print("fetched result: ", fetch_map)
