// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "./paddle_tensor.h"
#include "./paddle_tensor_impl.cu.h"
#include "./paddle_tensor_factory.cu.h"

namespace common {
template class CudaPaddleTensor<int64_t>;

std::shared_ptr<TensorAdapter<int64_t>>
CudaPaddleTensorFactory::create_int64_t(const std::vector<size_t> &shape) {
  auto ret = std::make_shared<CudaPaddleTensor<int64_t>>(_device_ctx);
  ret->reshape(shape);
  return ret;
}

std::shared_ptr<TensorAdapter<uint8_t>>
CudaPaddleTensorFactory::create_uint8_t(const std::vector<size_t> &shape) {
  auto ret = std::make_shared<CudaPaddleTensor<uint8_t>>(_device_ctx);
  ret->reshape(shape);
  return ret;
}

std::shared_ptr<TensorAdapter<int64_t>> CudaPaddleTensorFactory::create_int64_t() {
  auto ret = std::make_shared<CudaPaddleTensor<int64_t>>(_device_ctx);
  return ret;
}

} // namespace common
