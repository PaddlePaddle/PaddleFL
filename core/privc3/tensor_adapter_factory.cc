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

#include "tensor_adapter_factory.h"

namespace aby3 {

template <>
std::shared_ptr<TensorAdapter<int64_t>> TensorAdapterFactory::create() {
  return create_int64_t();
}

template <>
std::shared_ptr<TensorAdapter<int64_t>>
TensorAdapterFactory::create(const std::vector<size_t> &shape) {
  return create_int64_t(shape);
}

} // namespace aby3
