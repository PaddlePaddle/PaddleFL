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

#include <vector>

#include "paddle/fluid/framework/tensor_util.h"

#include "paddle_tensor.h"

namespace aby3 {

// slice_i = -1 indicate do not slice tensor
// otherwise vectorize tensor.Slice(slice_i, slice_i + 1)
template<typename T>
void TensorToVector(const TensorAdapter<T>* src, std::vector<T>* dst, int slice_i = -1) {
  auto& t = dynamic_cast<const PaddleTensor<T>*>(src)->tensor();
  if (slice_i == -1) {
    paddle::framework::TensorToVector(t, dst);
  } else {
    auto t_slice = t.Slice(slice_i, slice_i + 1);
     paddle::framework::TensorToVector(t_slice, dst);
  }
}

} // namespace aby3

