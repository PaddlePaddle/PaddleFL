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
