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
import redis
import logging
from . import table_base
from ..util import parse_bns_by_name


_LOGGER = logging.getLogger(__name__)


class RedisTable(table_base.TableBase):
    """
    redis client warpper with load balance
    """
    def __init__(self, ip_ports=None, bns_name=None):
        if ip_ports is not None and bns_name is not None:
            raise ValueError("can only accept one in ip_ports and bns_name")

        if ip_ports is not None:
            self.ip_ports_ = ip_ports
        elif bns_name is not None:
            self.ip_ports_ = parse_bns_by_name(bns_name=bns_name)
        else:
            raise TypeError("redis_table initializer need ip_port list")

        self.clients_ = self._build_connection_pool(self.ip_ports_)
        self.counter_ = 0

    def _build_connection_pool(self, ip_ports):
        """
        return redis connection pool
        """
        clients = list()
        for ip, port in ip_ports:
            clients.append(redis.StrictRedis(
                host=ip, port=port))  # connection pool
        return clients

    def _get_client(self):
        """
        get next client
        """
        client = self.clients_[self.counter_]
        self.counter_ = (self.counter_ + 1) % len(self.ip_ports_)
        return client

    def _set_value(self, data):
        client = self._get_client()
        with client.pipeline(transaction=False) as ppl:
            for k, v in data.items():
                ppl.set(k, v)
                ppl.execute()

    def _get_value(self, key):
        client = self._get_client()
        return client.get(key)

    def _get_values(self, keys):
        client = self._get_client()
        rnt = list()
        with client.pipeline(transaction=False) as ppl:
            for key in keys:
                ppl.get(key)
            rnt = ppl.execute()
        return rnt
