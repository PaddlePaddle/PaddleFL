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
random_port = 60001
current_ip = socket.gethostbyname(socket.gethostname())
print(current_ip)
os.system("python -m SimpleHTTPServer 8080 &")

#listening for client
context = zmq.Context()
zmq_socket = context.socket(zmq.REP)
zmq_socket.bind("tcp://{}:{}".format(current_ip, random_port))
print("binding tcp://{}:{}".format(current_ip, random_port))

#get mission and return the path of encoder
message = msgpack.loads(zmq_socket.recv())
print(message["mission"])
zmq_socket.send("user.py")

#wait client finish encoding
while True:
    message = zmq_socket.recv()
    if message == 'complete':
        break

#start training
os.system("python -u server.py > server.log &")
