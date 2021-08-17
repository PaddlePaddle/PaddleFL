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
import logging
import subprocess
import ujson as json
from . import table_base

_LOGGER = logging.getLogger(__name__)

class LocalTable(table_base.TableBase):

    def __init__(self, file_name):
        self.table = {}
        with open(file_name, 'r') as fin:
            for line in fin.readlines():
                cur_user = json.loads(line.strip())["user_info"]
                if "uid" in cur_user:
                    self.table[cur_user["uid"]] = cur_user
                else:
                    raise Exception("must have uid in user info")

    def _get_value(self, key):
        return self.table[key]

    def _get_values(self, keys):
        rnt = list()
        for key in keys:
            rnt.append(self.table[key])
        return rnt
