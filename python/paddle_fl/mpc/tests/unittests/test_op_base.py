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
Set base config for op unit tests.
"""
from multiprocessing import Pipe, Process
import os
import traceback
import unittest

import redis


class Aby3Process(Process):
    """
    Extends from Process, evaluate the computation party in aby3.
    """
    def __init__(self, *args, **kwargs):
        Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = Pipe()
        self._exception = None

    def run(self):
        """
        Override. Send any exceptions raised in
        subprocess to main process.
        """
        try:
            Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        """
        Get exception.
        """
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class TestOpBase(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(TestOpBase, self).__init__(methodName)
        # set redis server and port
        self.server = os.environ['TEST_REDIS_IP']
        self.port = os.environ['TEST_REDIS_PORT']
        self.party_num = 3

    def setUp(self):
        """
        Connect redis and delete all keys in all databases on the current host.
        :return:
        """
        r = redis.Redis(host=self.server, port=int(self.port))
        r.flushall()

    def multi_party_run(self, **kwargs):
        """
        Run 3 parties with target function or other additional arguments.
        :param kwargs:
        :return:
        """
        target = kwargs['target']

        parties = []
        for role in range(self.party_num):
            kwargs.update({'role': role})
            parties.append(Aby3Process(target=target, kwargs=kwargs))
            parties[-1].start()
        for party in parties:
            party.join()
            if party.exception:
                return party.exception
        return (True,)
