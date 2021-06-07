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

// Description:
// abstract mpc operation interface

#include "mpc_config.h"

namespace paddle {
namespace mpc {

const std::string MpcConfig::ROLE("role");
const std::string MpcConfig::NET_SIZE("net_size");
const std::string MpcConfig::LOCAL_ADDR("local.address");
const std::string MpcConfig::NET_SERVER_ADDR("net_server.address");
const std::string MpcConfig::NET_SERVER_PORT("net_server.port");
const std::string MpcConfig::ENDPOINTS("endpoints");
const std::string MpcConfig::NETWORK_MODE("network_mode");

const std::string MpcConfig::LOCAL_ADDR_DEFAULT("localhost");
const std::string MpcConfig::NET_SERVER_ADDR_DEFAULT("localhost");
const std::string MpcConfig::ENDPOINTS_DEFAULT("localhost:8900;localhost:8901;localhost:8902");
const std::string MpcConfig::NETWORK_MODE_DEFAULT("grpc");
const int MpcConfig::NET_SERVER_PORT_DEFAULT =
    6379; // default redis server port

} // mpc
} // paddle
