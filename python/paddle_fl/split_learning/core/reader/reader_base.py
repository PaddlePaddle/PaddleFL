# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

class ReaderBase(object):
    def __init__(self):
        pass

    def parse(self, db_value):
        raise NotImplementedError("Failed to parse db_value")

class TmpReader(ReaderBase):
    def __init__(self, place):
        super(FakeReader, self).__init__()
        self.place = place
    def parse(self, db_value):
        data_dict = {}
        data = {}
        data_dict["Host|input"] = np.random.randint(2, size=( 1, 1)).astype('int64')
        shapes = [[len(c) for c in data_dict["Host|input"]]]
        data["Host|input"] =  fluid.create_lod_tensor(data_dict["Host|input"].reshape(-1,1), shapes, self.place)
        data_dict["Customer|label"] = [1] #np.array([1]).astype('int64')
        data["Customer|label"] = data_dict["Customer|label"]
        return data
