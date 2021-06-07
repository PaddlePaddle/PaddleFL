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
// an PrivC protocol impl, including combination of operator, network and circuit
// context

#pragma once

#include "abstract_network.h"
#include "privc_operators.h"
#include "gloo/rendezvous/hash_store.h"
#include "network/mesh_network.h"
#include "mpc_operators.h"
#include "mpc_protocol.h"
#include "core/paddlefl_mpc/mpc_protocol/abstract_context.h"
#include "core/privc/privc_context.h"

namespace paddle {
namespace mpc {

using PrivCContext = privc::PrivCContext;

class PrivCProtocol : public MpcProtocol {
public:
  PrivCProtocol() : MpcProtocol("privc") {}
  // virtual ~PrivCProtocol() = default;

  void init(MpcConfig &config) override;
  
  // for test purpose
  void init_with_store(const MpcConfig &config,
                       std::shared_ptr<gloo::rendezvous::Store> store) override;

  std::shared_ptr<MpcOperators> mpc_operators() override;

  std::shared_ptr<AbstractNetwork> network() override;

  std::shared_ptr<AbstractContext> mpc_context() override;

private:
  bool _is_initialized = false;
  const size_t net_size = 2;
  const std::string PROT_INIT_ERR = "The protocol is not yet initialized.";
  std::shared_ptr<MpcOperators> _operators;
  std::shared_ptr<AbstractNetwork> _network;
  std::shared_ptr<AbstractContext> _circuit_ctx;
};

} // mpc
} // paddle
