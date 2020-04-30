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

// Description: A paddle_encrypted executor for running a mpc program

#pragma once
#include "gloo/rendezvous/hash_store.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_config.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_protocol.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_protocol_factory.h"
#include <memory>
#include <mutex>

namespace paddle {
namespace mpc {

class MpcInstance {
private:
  MpcInstance(const std::string &protocol_name, const MpcConfig &config)
      : _protocol_name(protocol_name), _mpc_config(config) {}

  // for test purpose
  void prepare_mpc_protocol_with_store(
      std::shared_ptr<gloo::rendezvous::Store> store) {
    _s_mpc_protocol = MpcProtocolFactory::build(_protocol_name);
    PADDLE_ENFORCE_NOT_NULL(_s_mpc_protocol, "Unrecognized mpc protocol: %s",
                            _protocol_name);
    _s_mpc_protocol->init_with_store(_mpc_config, store);
  }

  void prepare_mpc_protocol() {
    _s_mpc_protocol = MpcProtocolFactory::build(_protocol_name);
    PADDLE_ENFORCE_NOT_NULL(_s_mpc_protocol, "Unrecognized mpc protocol: %s",
                            _protocol_name);
    _s_mpc_protocol->init(_mpc_config);
  }

  static void init_mpc(const std::string &protocol_name,
                       const MpcConfig &mpc_config) {
    _s_mpc_instance.reset(new MpcInstance(protocol_name, mpc_config));
    _s_mpc_instance->prepare_mpc_protocol();
  }

  // for test purpose
  static void
  init_mpc_with_store(const std::string &protocol_name,
                      const MpcConfig &mpc_config,
                      std::shared_ptr<gloo::rendezvous::Store> store) {
    _s_mpc_instance.reset(new MpcInstance(protocol_name, mpc_config));
    _s_mpc_instance->prepare_mpc_protocol_with_store(store);
  }

public:
  static std::shared_ptr<MpcInstance>
  init_instance(const std::string &protocol_name, const MpcConfig &mpc_config) {
    std::call_once(_s_init_flag, &MpcInstance::init_mpc, protocol_name,
                   mpc_config);
    return _s_mpc_instance;
  }

  // for test purpose
  static std::shared_ptr<MpcInstance>
  init_instance_with_store(const std::string &protocol_name,
                           const MpcConfig &mpc_config,
                           std::shared_ptr<gloo::rendezvous::Store> store) {
    std::call_once(_s_init_flag, &MpcInstance::init_mpc_with_store,
                   protocol_name, mpc_config, store);
    return _s_mpc_instance;
  }

  static std::shared_ptr<MpcInstance> mpc_instance() {
    PADDLE_ENFORCE_NOT_NULL(_s_mpc_instance,
                            "Mpc instance is not initialized!");
    return _s_mpc_instance;
  }

  static std::shared_ptr<MpcProtocol> mpc_protocol() {
    PADDLE_ENFORCE_NOT_NULL(_s_mpc_protocol, "MpcProtocol is null.");
    return _s_mpc_protocol;
  }

private:
  static thread_local std::once_flag _s_init_flag;
  const std::string _protocol_name;
  MpcConfig _mpc_config;
  static thread_local std::shared_ptr<MpcInstance> _s_mpc_instance;
  static thread_local std::shared_ptr<MpcProtocol> _s_mpc_protocol;
};

} // namespace framework
} // namespace paddle
