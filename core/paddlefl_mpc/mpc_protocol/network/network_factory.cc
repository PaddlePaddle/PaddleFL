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
// limitations under the License.nguage governing permissions and

#include "network_factory.h"

#include "gloo/rendezvous/redis_store.h"
#include "paddle/fluid/framework/tensor.h"

#include "core/paddlefl_mpc/mpc_protocol/network/mesh_network.h"

#ifdef WITH_GRPC
#include "core/paddlefl_mpc/mpc_protocol/network/mesh_network_grpc.h"
#endif

namespace paddle {
namespace mpc {

void MpcNetworkFactory::register_creator() {
    if (!_is_initialized) {

        auto gloo_net_creator = [](const MpcConfig &config) {
            auto party_id = config.get_int(MpcConfig::ROLE);
            auto net_size = config.get_int(MpcConfig::NET_SIZE);
            auto local_addr = config.get(MpcConfig::LOCAL_ADDR, MpcConfig::LOCAL_ADDR_DEFAULT);
            std::string store_prefix = "Paddle-mpc";
            auto server_addr = config.get(MpcConfig::NET_SERVER_ADDR,
                                          MpcConfig::NET_SERVER_ADDR_DEFAULT);
            auto server_port = config.get_int(MpcConfig::NET_SERVER_PORT,
                                              MpcConfig::NET_SERVER_PORT_DEFAULT);
            auto store = std::make_shared<gloo::rendezvous::RedisStore>(server_addr, server_port);

            return std::make_shared<MeshNetwork>(party_id, local_addr, net_size, store_prefix, store);
        };

        _creator_map.insert({"gloo", gloo_net_creator});

#ifdef WITH_GRPC
        auto grpc_net_creator = [](const MpcConfig &config) {
            auto party_id = config.get_int(MpcConfig::ROLE);
            auto net_size = config.get_int(MpcConfig::NET_SIZE);
            auto endpoints = config.get(MpcConfig::ENDPOINTS, MpcConfig::ENDPOINTS_DEFAULT);

            return std::make_shared<MeshNetworkGrpc>(party_id, net_size, endpoints);
        };

        _creator_map.insert({"grpc", grpc_net_creator});
#endif

    }
    _is_initialized = true;
}

MpcNetworkFactory::Creator MpcNetworkFactory::get_creator(const std::string &name) {
    if (!_is_initialized) {
        register_creator();
    }
    return _creator_map[to_lowercase(name)];
}

MpcNetworkFactory::CreatorMap MpcNetworkFactory::_creator_map;
bool MpcNetworkFactory::_is_initialized = false;

} // mpc
} // paddle
