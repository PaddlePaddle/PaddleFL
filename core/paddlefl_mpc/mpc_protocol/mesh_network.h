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
#include <string>

#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/hash_store.h"

#include "abstract_network.h"

namespace paddle {
namespace mpc {

// A full-connected network based on underlying GLOO toolkit, with the network
// size of 3
class MeshNetwork : public paddle::mpc::AbstractNetwork {
public:
  // a ctor called for the explicit netowrk size and store as parameters
  // prefix: the prefix of keys in the store to differentiate different runs
  // example:
  //     auto store = std::make_shared<gloo::rendezvous::HashStore>();
  // (in each thread:)
  //     paddle::mpc::MeshNetwork net(0, "127.0.0.1", 3, "test_prefix", store);
  //     net.init();
  //     net.send(1, data, sizeof(data))
  //
  MeshNetwork(const size_t party_id, const std::string &local_addr,
              const size_t net_size, const std::string &prefix,
              std::shared_ptr<gloo::rendezvous::Store> store)
      : _party_id(party_id), _local_addr(local_addr), _net_size(net_size),
        _store_prefix(prefix), _store(std::move(store)),
        _is_initialized(false) {}

  virtual ~MeshNetwork() = default;

  void send(size_t party, const void *data, size_t size) override;

  void recv(size_t party, void *data, size_t size) override;

  size_t party_id() const override { return _party_id; };

  size_t party_num() const override { return _net_size; };

  // must be called before use
  void init();

private:
  const size_t _party_id;
  const size_t _net_size;
  const std::string _local_addr;
  const std::string _store_prefix;

  std::shared_ptr<gloo::rendezvous::Store> _store;
  std::shared_ptr<gloo::rendezvous::Context> _rendezvous_ctx;

  bool _is_initialized;
};

} // mpc
} // paddle
