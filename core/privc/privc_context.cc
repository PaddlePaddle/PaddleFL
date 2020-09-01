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

#include "core/privc/triplet_generator.h"
#include "core/privc/privc_context.h"
namespace privc {

void PrivCContext::set_triplet_generator(std::shared_ptr<TripletGenerator<int64_t, SCALING_N>>& tripletor) {
    _tripletor = tripletor;
}

std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> PrivCContext::triplet_generator() {
  PADDLE_ENFORCE_NE(_tripletor, nullptr, "must set triplet generator first.");
  return _tripletor;
}

} // namespace privc
