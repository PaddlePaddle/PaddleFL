
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
This module provide one hot encoding tools, implemented by OT (Oblivious Transfer)-based
PSI (Private Set Intersection) algorithm.
"""
from multiprocessing.connection import Client, Listener
import mpc_data_utils as mdu

__all__ = ['one_hot_encoding_map', ]


def one_hot_encoding_map(input_set, host_addr, is_client=True):
    """
    A protocol to get agreement between 2 parties for encoding one
    discrete feature to one hot vector via OT-PSI.

    Args:
        input_set (set:str): The set of possible feature value owned by this
        party. Element of set is str, convert before pass in.

        host_addr (str): The info of host_addr,e.g., ip:port

        is_receiver (bool): True if this party plays as socket client
        otherwise, plays as socket server

    Return Val: dict, int.
        dict key: feature values in input_set,
        dict value: corresponding idx in one hot vector.

        int: length of one hot vector for this feature.

    Examples:
        .. code-block:: python

            import paddle_fl.mpc.data_utils
            import sys

            is_client = sys.argv[1] == "1"

            a = set([str(x) for x in range(7)])
            b = set([str(x) for x in range(5, 10)])

            addr = "127.0.0.1:33784"

            ins = a if is_client else b

            x, y = paddle_fl.mpc.data_utils.one_hot_encoding_map(ins, addr, is_client)
            # y = 10
            # x['5'] = 0, x['6'] = 1
            # for those feature val owned only by one party, dict val shall
            not be conflicting.
            print(x, y)
    """

    ip = host_addr.split(":")[0]
    port = int(host_addr.split(":")[1])

    if is_client:
        intersection = input_set
        intersection = mdu.recv_psi(ip, port, intersection)
        intersection = sorted(list(intersection))
        # Only the receiver can obtain the result.
        # Send result to other parties.
    else:
        ret_code = mdu.send_psi(port, input_set)
        if ret_code != 0:
            raise RuntimeError("Errors occurred in PSI send lib, "
                               "error code = {}".format(ret_code))

    if not is_client:
        server = Listener((ip, port))
    conn = Client((ip, port)) if is_client else server.accept()


    if is_client:
        conn.send(intersection)

        diff_size_local = len(input_set) - len(intersection)
        conn.send(diff_size_local)
        diff_size_remote = conn.recv()

    else:
        intersection = conn.recv()

        diff_size_local = len(input_set) - len(intersection)

        diff_size_remote = conn.recv()
        conn.send(diff_size_local)

    conn.close()
    if not is_client:
        server.close()

    ret = dict()

    cnt = 0

    for x in intersection:
        ret[x] = cnt
        cnt += 1
    if is_client:
        cnt += diff_size_remote
    for x in [x for x in input_set if x not in intersection]:
        ret[x] = cnt
        cnt += 1

    return ret, len(intersection) + diff_size_local + diff_size_remote
