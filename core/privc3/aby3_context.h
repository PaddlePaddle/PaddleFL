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

#include <algorithm>
#include <memory>

#include "core/paddlefl_mpc/mpc_protocol/abstract_context.h"
#include "core/paddlefl_mpc/mpc_protocol/abstract_network.h"
#include "prng_utils.h"

namespace aby3 {

using AbstractNetwork = paddle::mpc::AbstractNetwork;
using AbstractContext = paddle::mpc::AbstractContext;

class ABY3Context : public AbstractContext {
public:
  ABY3Context(size_t party, std::shared_ptr<AbstractNetwork> network,
                 block seed = g_zero_block,
                 block seed2 = g_zero_block) :
                 AbstractContext::AbstractContext(party, network) {
    set_num_party(3);

    if (common::equals(seed, common::g_zero_block)) {
      seed = common::block_from_dev_urandom();
    }

    if (common::equals(seed2, common::g_zero_block)) {
      seed2 = common::block_from_dev_urandom();
    }
    set_random_seed(seed, 0);
    // seed2 is private
    set_random_seed(seed2, 2);

    // 3 for 3-party computation
    size_t party_pre = pre_party();
    size_t party_next = next_party();

    if (party == 1) {
      block recv_seed = this->network()->template recv<block>(party_next);
      this->network()->template send(party_pre, seed);
      seed = recv_seed;
    } else {
      this->network()->template send(party_pre, seed);
      seed = this->network()->template recv<block>(party_next);
    }

    set_random_seed(seed, 1);
  }

  ABY3Context(const ABY3Context &other) = delete;

  ABY3Context &operator=(const ABY3Context &other) = delete;
protected:
  PseudorandomNumberGenerator& get_prng(size_t idx) override {
    return _prng[idx];
  }
private:
  PseudorandomNumberGenerator _prng[3];
};

} // namespace aby3
