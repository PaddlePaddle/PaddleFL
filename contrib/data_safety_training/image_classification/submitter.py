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
