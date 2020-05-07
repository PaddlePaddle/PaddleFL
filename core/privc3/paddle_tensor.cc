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

#include "paddle_tensor.h"

namespace aby3 {

std::shared_ptr<TensorAdapter<int64_t>>
PaddleTensorFactory::create_int64_t(const std::vector<size_t> &shape) {
  auto ret = std::make_shared<PaddleTensor<int64_t>>(_device_ctx);
  ret->reshape(shape);
  return ret;
}

std::shared_ptr<TensorAdapter<int64_t>> PaddleTensorFactory::create_int64_t() {
  auto ret = std::make_shared<PaddleTensor<int64_t>>(_device_ctx);
  return ret;
}

} // namespace aby3
