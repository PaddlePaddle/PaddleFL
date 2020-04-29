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
// mpc protocol base class

#pragma once

#include "abstract_network.h"
#include "gloo/rendezvous/hash_store.h"
#include "mpc_config.h"
#include "mpc_operators.h"
#include "privc3/circuit_context.h"

namespace paddle {
namespace mpc {

class MpcProtocol {
public:
  MpcProtocol(const std::string &name) : _name(name){};
  virtual ~MpcProtocol() = default;

  virtual std::string name() const { return _name; }

  virtual void init(const MpcConfig &config) = 0;

  // for test purpose
  virtual void
  init_with_store(const MpcConfig &config,
                  std::shared_ptr<gloo::rendezvous::Store> store) = 0;

  virtual std::shared_ptr<MpcOperators> mpc_operators() = 0;

  virtual std::shared_ptr<AbstractNetwork> network() = 0;

  virtual std::shared_ptr<aby3::CircuitContext> mpc_context() = 0;

private:
  const std::string _name;
};

} // mpc
} // paddle
