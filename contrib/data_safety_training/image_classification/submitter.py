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

import zmq
import socket
import msgpack
import os
mission_dict = {"mission": "image classification", "image_size": [3, 32, 32]}
#send request
context = zmq.Context()
zmq_socket = context.socket(zmq.REQ)
zmq_socket.connect("tcp://127.0.0.1:60001")
zmq_socket.send(msgpack.dumps(mission_dict))

#get and download encoder
file = zmq_socket.recv()
os.system("wget 127.0.0.1:8080/{}".format(file))

#data encoding
os.system("python -u user.py > user.log")
zmq_socket.send("complete")
