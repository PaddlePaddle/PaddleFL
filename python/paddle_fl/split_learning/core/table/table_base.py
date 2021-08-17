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


class TableBase(object):
    def __init__(self):
        pass

    def _get_value(self, key):
        """
        args: str key
        return: raw bytestream data
        """
        raise NotImplementedError("Failed to get value.(key: {})".format(key))

    def _get_values(self, keys):
        """
        args: list of str key
        return: raw bytestream data list
        """
        raise NotImplementedError("Failed to get value.(key: {})".format(keys))

    def _set_value(self, data):
        """
        args: dict of data
        """
        raise NotImplementedError("Failed to set value.")

    def insert(self, data):
        if isinstance(data, dict):
            self._set_value(data)
        else:
            raise TypeError("Cannot resolve {} type".format(type(data)))

    def lookup(self, key):
        if isinstance(key, list):
            return self._get_values(key)
        elif isinstance(key, str):
            return self._get_value(key)
        else:
            raise TypeError("Cannot resolve {} type".format(type(key)))
