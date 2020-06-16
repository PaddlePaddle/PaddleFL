// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace psi {

class NetIO {
public:
  NetIO(const NetIO &rhs) = delete;

  NetIO &operator=(const NetIO &rhs) = delete;

  NetIO(const char *address, int port, bool quiet = false, int timeout_s = 0,
        bool manual_accept = 0) {
    _quiet = quiet;
    _port = port;
    _is_server = (address == nullptr);
    if (address == nullptr) {
      struct sockaddr_in serv;
      memset(&serv, 0, sizeof(serv));
      serv.sin_family = AF_INET;
      // set our address to any interface
      serv.sin_addr.s_addr = htonl(INADDR_ANY);
      // set the server port number
      serv.sin_port = htons(port);
      _mysocket = socket(AF_INET, SOCK_STREAM, 0);

      int reuse = 1;
      setsockopt(_mysocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,
                 sizeof(reuse));

      if (bind(_mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) <
          0) {
        throw std::runtime_error("socket error: fail to bind port " +
                                 std::to_string(port) + ", errno: " +
                                 std::to_string(errno));
      }
      if (listen(_mysocket, 1) < 0) {
        throw std::runtime_error("socket error: fail to listen on port " +
                                 std::to_string(port) + ", errno: " +
                                 std::to_string(errno));
      }
      if (!manual_accept) {
        accept();
      }
    } else {
      _addr = std::string(address);

      struct sockaddr_in dest;
      memset(&dest, 0, sizeof(dest));
      dest.sin_family = AF_INET;
      dest.sin_addr.s_addr = inet_addr(address);
      dest.sin_port = htons(port);

      while (1) {
        _consocket = socket(AF_INET, SOCK_STREAM, 0);

        if (connect(_consocket, (struct sockaddr *)&dest,
                    sizeof(struct sockaddr)) == 0) {
          break;
        }

        close(_consocket);
        usleep(1000);
      }
    }
    set_nodelay();

    _timeout = false;
    if (timeout_s) {
      set_recv_timeout(timeout_s);
      _timeout = true;
    }

    if (!_quiet) {
      std::cout << "connected\n";
    }
  }

  virtual ~NetIO() {
    close(_consocket);
    if (_is_server) {
      close(_mysocket);
    }
  }

  void accept() {
    struct sockaddr_in dest;
    socklen_t socksize = sizeof(struct sockaddr_in);
    _consocket = ::accept(_mysocket, reinterpret_cast<struct sockaddr *>(&dest),
                          &socksize);
  }

  bool connected() const { return _consocket != -1; }

  void send_data(const void *data, size_t len) {

    for (; len;) {
      int ret = send(_consocket, data, len, 0);
      if (ret < 0) {
        throw std::runtime_error("socket error: send, errno: " +
                                 std::to_string(errno));
      }
      data = reinterpret_cast<const char *>(data) + ret;
      len -= ret;
    }
  }

  void recv_data(void *data, size_t len) {

    for (size_t recved = 0; recved < len;) {
      ssize_t ret =
          recv(_consocket, (char *)data + recved, len - recved, MSG_WAITALL);
      if (ret < 0) {
        throw std::runtime_error("socket error: recv, errno: " +
                                 std::to_string(errno));
      } else if (ret == 0) {
        throw std::runtime_error("socket error: 0 byte recved, "
                                 "socket shutdown by peer");
      }
      recved += ret;
    }
  }

  // recv with timeout, set timeout first
  void recv_data_with_timeout(void *data, size_t len) {
    if (!_timeout) {
      recv_data(data, len);
      return;
    }
    ssize_t ret = recv(_consocket, (char *)data, len, MSG_WAITALL);
    if (ret == -1 && errno != EAGAIN) {
      throw std::runtime_error("socket error: recv, errno: " +
                               std::to_string(errno));
    }
    if (ret != (ssize_t)len) {
      throw std::runtime_error("socket error: recv timeout");
    }
    return;
  }

private:
  bool _is_server;
  int _mysocket = -1;
  int _consocket = -1;
  std::string _addr;
  int _port;
  bool _quiet;
  bool _timeout;

  void set_nodelay() {
    const int one = 1;
    setsockopt(_consocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }

  void set_non_blocking() {
    int flags = fcntl(_consocket, F_GETFL, 0);
    fcntl(_consocket, F_SETFL, flags | O_NONBLOCK);
  }

  void set_recv_timeout(int timeout_s) {
    struct timeval tv;
    tv.tv_sec = timeout_s;
    tv.tv_usec = 0;
    setsockopt(_consocket, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv,
               sizeof(tv));
  }
};
} // namespace psi
