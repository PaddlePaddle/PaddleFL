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

#include "core/paddlefl_mpc/mpc_protocol/network/mesh_network_grpc.h"

#include <thread>

#include "gtest/gtest.h"

namespace paddle {
namespace mpc {

class NetworkTest : public ::testing::Test {
public:

    std::string _endpoints;

    std::shared_ptr<AbstractNetwork> _p0;
    std::shared_ptr<AbstractNetwork> _p1;
    std::shared_ptr<AbstractNetwork> _p2;

    std::shared_ptr<MeshNetworkGrpc> _n0;
    std::shared_ptr<MeshNetworkGrpc> _n1;
    std::shared_ptr<MeshNetworkGrpc> _n2;

    NetworkTest() : _endpoints("localhost:8900;localhost:8901;localhost:8902") {
        _n0 = std::make_shared<MeshNetworkGrpc>(0, 3, _endpoints);
        _n1 = std::make_shared<MeshNetworkGrpc>(1, 3, _endpoints);
        _n2 = std::make_shared<MeshNetworkGrpc>(2, 3, _endpoints);

        _p0 = _n0;
        _p1 = _n1;
        _p2 = _n2;
    }

    void SetUp() {
        std::thread t0([this]() { _n0->init(); });
        std::thread t1([this]() { _n1->init(); });
        std::thread t2([this]() { _n2->init(); });

        t0.join();
        t1.join();
        t2.join();
    }
};

TEST_F(NetworkTest, basic_test) {
    int buf[3] = {0, 1, 2};
    std::thread t0([this, &buf]() {
        _p0->template send(1, buf[0]);
        buf[0] = _p0->template recv<int>(2);
    });

    std::thread t1([this, &buf]() {
        int to_send = buf[1];
        buf[1] = _p1->template recv<int>(0);
        _p1->template send(2, to_send);
    });

    std::thread t2([this, &buf]() {
        int to_send = buf[2];
        buf[2] = _p2->template recv<int>(1);
        _p2->template send(0, to_send);
    });

    t0.join();
    t1.join();
    t2.join();

    EXPECT_EQ(2, buf[0]);
    EXPECT_EQ(0, buf[1]);
    EXPECT_EQ(1, buf[2]);
}

} // namespace mpc
} // namespace paddle
