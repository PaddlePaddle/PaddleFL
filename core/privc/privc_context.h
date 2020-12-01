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
#include "core/common/prng.h"
#include "core/common/rand_utils.h"

namespace privc {

using AbstractNetwork = paddle::mpc::AbstractNetwork;
using AbstractContext = paddle::mpc::AbstractContext;
using block = common::block;

static const size_t SCALING_N = 32;
template <typename T, size_t N>
class TripletGenerator;

class PrivCContext : public AbstractContext {
public:
  PrivCContext(size_t party, std::shared_ptr<AbstractNetwork> network,
                 block seed = common::g_zero_block);

  PrivCContext(const PrivCContext &other) = delete;

  PrivCContext &operator=(const PrivCContext &other) = delete;

  std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> triplet_generator();

protected:
  common::PseudorandomNumberGenerator& get_prng(size_t idx) override {
    return _prng;
  }

private:
  std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> _tripletor;
  common::PseudorandomNumberGenerator _prng;
};

} // namespace privc
