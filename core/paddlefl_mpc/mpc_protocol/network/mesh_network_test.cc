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

#include "core/paddlefl_mpc/mpc_protocol/network/mesh_network.h"

#include <thread>

#include "gtest/gtest.h"

namespace paddle {
namespace mpc {

class NetworkTest : public ::testing::Test {
public:
  std::string _addr;

  std::string _prefix;

  std::shared_ptr<gloo::rendezvous::HashStore> _store;

  MeshNetwork _n0;
  MeshNetwork _n1;

  AbstractNetwork *_p0;
  AbstractNetwork *_p1;

  NetworkTest()
      : _addr("127.0.0.1"), _prefix("test_prefix"),
        _store(std::make_shared<gloo::rendezvous::HashStore>()),
        _n0(0, _addr, 2, _prefix, _store), _n1(1, _addr, 2, _prefix, _store),
        _p0(&_n0), _p1(&_n1) {}

  void SetUp() {
    std::thread t0([this]() { _n0.init(); });
    std::thread t1([this]() { _n1.init(); });

    t0.join();
    t1.join();
  }
};

TEST_F(NetworkTest, basic_test) {
  int buf[2] = {0, 1};
  std::thread t0([this, &buf]() {
    _p0->template send(1, buf[0]);
    buf[0] = _p0->template recv<int>(1);
  });

  std::thread t1([this, &buf]() {
    int to_send = buf[1];
    buf[1] = _p1->template recv<int>(0);
    _p1->template send(0, to_send);
  });
  t0.join();
  t1.join();

  EXPECT_EQ(1, buf[0]);
  EXPECT_EQ(0, buf[1]);
}

} // namespace mpc
} // namespace paddle
