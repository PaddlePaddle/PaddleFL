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
#ifdef WTIH_GRPC
#include "core/paddlefl_mpc/mpc_protocol/network/mesh_network_grpc.h"
#endif

#ifdef USE_CUDA
#include "./nccl_network.h"
#include "core/psi/net_io.h"
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

#ifdef WTIH_GRPC
        auto grpc_net_creator = [](const MpcConfig &config) {
            auto party_id = config.get_int(MpcConfig::ROLE);
            auto net_size = config.get_int(MpcConfig::NET_SIZE);
            auto endpoints = config.get(MpcConfig::ENDPOINTS, MpcConfig::ENDPOINTS_DEFAULT);

            return std::make_shared<MeshNetworkGrpc>(party_id, net_size, endpoints);
        };

        _creator_map.insert({"grpc", grpc_net_creator});
#endif

#ifdef USE_CUDA
        auto nccl_net_creator = [](const MpcConfig &config) {
            auto party_id = config.get_int(MpcConfig::ROLE);
            auto net_size = config.get_int(MpcConfig::NET_SIZE);
            auto server_addr = config.get(MpcConfig::NET_SERVER_ADDR,
                                          MpcConfig::NET_SERVER_ADDR_DEFAULT);
            // reuse endpoints for ports
            auto endpoints = config.get(MpcConfig::ENDPOINTS, MpcConfig::ENDPOINTS_DEFAULT);

            std::vector<std::string> ports;

            std::istringstream stream(endpoints);
            for (std::string each; std::getline(stream, each, ','); ports.emplace_back(std::move(each))) {}

            ncclUniqueId id;
            if (party_id) {
                psi::NetIO io(server_addr.c_str(), std::stol(ports[party_id - 1]), true);
                io.recv_data(&id, sizeof(id));
            } else { // party 0
                id = NcclNetwork::get_nccl_id();
                for (int i = 0; i < net_size - 1; ++i) {
                    psi::NetIO io(nullptr, std::stol(ports[i]), true);
                    io.send_data(&id, sizeof(id));
                }
            }

            auto dev_id = config.get_int(MpcConfig::DEVICE_ID);

            LOG(WARNING) << "init nccl with stream of CUDA device " << dev_id;

            paddle::platform::CUDAPlace gpu(dev_id);
            auto& pool = paddle::platform::DeviceContextPool::Instance();
            auto* dev_ctx = pool.template GetByPlace<paddle::platform::CUDAPlace>(gpu);
            return std::make_shared<NcclNetwork>(party_id, net_size, id, dev_ctx->stream());
        };
        _creator_map.insert({"nccl", nccl_net_creator});
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
