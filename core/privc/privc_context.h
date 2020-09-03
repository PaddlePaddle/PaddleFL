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
#include "core/privc3/prng_utils.h"

namespace privc {

using AbstractNetwork = paddle::mpc::AbstractNetwork;
using AbstractContext = paddle::mpc::AbstractContext;
using block = psi::block;

static const size_t SCALING_N = 32;
template <typename T, size_t N>
class TripletGenerator;

class PrivCContext : public AbstractContext {
public:
  PrivCContext(size_t party, std::shared_ptr<AbstractNetwork> network,
                 block seed = psi::g_zero_block):
                 AbstractContext::AbstractContext(party, network), _tripletor{nullptr} {
    set_num_party(2);

    if (psi::equals(seed, psi::g_zero_block)) {
      seed = psi::block_from_dev_urandom();
    }
    set_random_seed(seed, 0);
  }

  PrivCContext(const PrivCContext &other) = delete;

  PrivCContext &operator=(const PrivCContext &other) = delete;

  void set_triplet_generator(std::shared_ptr<TripletGenerator<int64_t, SCALING_N>>& tripletor);

  std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> triplet_generator();

protected:
  psi::PseudorandomNumberGenerator& get_prng(size_t idx) override {
    return _prng;
  }

private:
  std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> _tripletor;
  psi::PseudorandomNumberGenerator _prng;
};

} // namespace privc
