#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
This module provide data alignment tools, implemented by OT (Oblivious Transfer)-based
PSI (Private Set Intersection) algorithm.
"""
import os
import sys
import mpc_data_utils as mdu
from multiprocessing.connection import Client, Listener
__all__ = ['align', ]


def align(input_set, party_id, endpoints, is_receiver=True):
    """
    Align the data owned by each data party.
    :param input_set: set. The id set of input data owned by this
    party.

    :param party_id: int. The id of this data party, which is
    natural number named from 0.
    :param endpoints: str. The info of all data parties,e.g.,
    id1:ip1:port1,id2:ip2:port2
    :param is_receiver: bool. True if this data party is a receiver
    role among all parties. Note that there is only one receiver
    who can obtain the result of aligning and then send it to other
    parties.
    :return: set. The intersection of data id.
    """
    all_parties = endpoints.split(",")
    _party_idx = _find_party_idx(party_id, all_parties)
    if _party_idx < 0:
        raise RuntimeError("Could not find endpoint with id: {}".format(
            party_id))

    if is_receiver:
        del (all_parties[_party_idx])
        senders = all_parties
        result = input_set
        for sender in senders:
            ip_addr = sender.split(":")[1]
            port = int(sender.split(":")[2])
            result = mdu.recv_psi(ip_addr, port, result)
            result = set(result)
        # Only the receiver can obtain the result.
        # Send result to other parties.
        _send_align_result(result, senders)
    else:
        sender = all_parties[_party_idx]
        port = int(sender.split(":")[2])
        ret_code = mdu.send_psi(port, input_set)
        if ret_code != 0:
            raise RuntimeError("Errors occurred in PSI send lib, "
                               "error code = {}".format(ret_code))
        result = _recv_align_result(sender)
    return result


def _find_party_idx(party_id, endpoint_list):
    """
    return the index of the given party id in the endpoint list
    :param party_id: party id
    :param endpoint_list: list of endpoints
    :return: the index of endpoint with the party_id, or -1 if not found
    """
    for idx in range(0, len(endpoint_list)):
        if party_id == int(endpoint_list[idx].split(":")[0]):
            return idx
    return -1


def _send_align_result(result, send_list):
    """
    Send align result to other data parties. This is used by the
    receiver when align.

    :param result: set. The align result.
    :param send_list: list. The data parties who receive the result.
    Each party is represented as "id:ip:port".
    :return:
    """
    for host in send_list:
        ip_addr = host.split(":")[1]
        port = int(host.split(":")[2])
        client = Client((ip_addr, port))
        client.send(result)
        client.close()


def _recv_align_result(host):
    """
    Receive align result from receiver.

    :param host: str. The host who is waiting for align result.
    The host is represented as "id:ip:port".
    :return: set. The received align result.
    """
    ip_addr = host.split(":")[1]
    port = int(host.split(":")[2])
    server = Listener((ip_addr, port))
    conn = server.accept()
    result = conn.recv()
    conn.close()
    server.close()
    return result
