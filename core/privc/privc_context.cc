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
#include <algorithm>
#include <memory>

#include "core/privc/privc_context.h"
#include "core/privc/he_triplet.h"
#include "core/privc/ot.h"

namespace privc {

PrivCContext::PrivCContext(size_t party, std::shared_ptr<AbstractNetwork> network,
                block seed):
                AbstractContext::AbstractContext(party, network) {
  set_num_party(2);

  if (common::equals(seed, common::g_zero_block)) {
    seed = common::block_from_dev_urandom();
  }
  set_random_seed(seed, 0);
  auto garbled_delta = this->template gen_random_private<block>();
  auto ot_base_choice = this->template gen_random_private<block>();
  _ot = std::make_shared<OT>(
                    ot_base_choice,
                    garbled_delta,
                    this->network(),
                    this->party(),
                    this->next_party());
  _ot->init();
  _tripletor = std::make_shared<HETriplet<uint64_t, PRIVC_FIXED_POINT_SCALING_FACTOR>>(
                                                this->party(),
                                                this->network(),
                                                _prng);
  _tripletor->init();
}

std::shared_ptr<HETriplet<uint64_t, PRIVC_FIXED_POINT_SCALING_FACTOR>> PrivCContext::triplet_generator() {
  PADDLE_ENFORCE_NE(_tripletor, nullptr, "must set triplet generator first.");
  return _tripletor;
}

std::shared_ptr<OT>& PrivCContext::ot() {
  return _ot;
}

void PrivCContext::set_triplet_generator(std::shared_ptr<HETriplet<uint64_t, PRIVC_FIXED_POINT_SCALING_FACTOR>> tripletor) {
  _tripletor = tripletor;
}

} // namespace privc
