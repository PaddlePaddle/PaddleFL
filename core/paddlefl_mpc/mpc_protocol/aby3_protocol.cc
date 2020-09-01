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
// an ABY3 protocol impl, including combination of operator, network and circuit
// context

#include "aby3_protocol.h"
#include "gloo/rendezvous/redis_store.h"
#include "mpc_protocol_factory.h"

namespace paddle {
namespace mpc {

void Aby3Protocol::init_with_store(
    const MpcConfig &config, std::shared_ptr<gloo::rendezvous::Store> store) {
  if (_is_initialized) {
    return;
  }

  PADDLE_ENFORCE_NOT_NULL(store);

  // read role, address and other info
  auto role = config.get_int(Aby3Config::ROLE);
  PADDLE_ENFORCE_LT(role, 3, "Input role should be less than party_size(3).");

  auto local_addr =
      config.get(Aby3Config::LOCAL_ADDR, Aby3Config::LOCAL_ADDR_DEFAULT);
  auto net_server_addr = config.get(Aby3Config::NET_SERVER_ADDR,
                                    Aby3Config::NET_SERVER_ADDR_DEFAULT);
  auto net_server_port = config.get_int(Aby3Config::NET_SERVER_PORT,
                                        Aby3Config::NET_SERVER_PORT_DEFAULT);

  auto mesh_net = std::make_shared<MeshNetwork>(
      role, local_addr, 3 /* netsize */, "Paddle-mpc" /* key-prefix in store*/,
      store);
  mesh_net->init();

  _network = std::move(mesh_net);
  _circuit_ctx = std::make_shared<ABY3Context>(role, _network);
  _operators = std::make_shared<Aby3OperatorsImpl>();
  _is_initialized = true;
}

std::shared_ptr<MpcOperators> Aby3Protocol::mpc_operators() {
  PADDLE_ENFORCE(_is_initialized, PROT_INIT_ERR);
  return _operators;
}

std::shared_ptr<AbstractNetwork> Aby3Protocol::network() {
  PADDLE_ENFORCE(_is_initialized, PROT_INIT_ERR);
  return _network;
}

std::shared_ptr<AbstractContext> Aby3Protocol::mpc_context() {
  PADDLE_ENFORCE(_is_initialized, PROT_INIT_ERR);
  return _circuit_ctx;
}

void Aby3Protocol::init(const MpcConfig &config) {
  if (_is_initialized) {
    return;
  }

  auto server_addr = config.get(Aby3Config::NET_SERVER_ADDR,
                                Aby3Config::NET_SERVER_ADDR_DEFAULT);
  auto server_port = config.get_int(Aby3Config::NET_SERVER_PORT,
                                    Aby3Config::NET_SERVER_PORT_DEFAULT);
  auto gloo_store =
      std::make_shared<gloo::rendezvous::RedisStore>(server_addr, server_port);

  init_with_store(config, gloo_store);
}

} // mpc
} // paddle
