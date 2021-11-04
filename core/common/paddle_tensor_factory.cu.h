
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

#pragma once

#include "./tensor_adapter_factory.h"

namespace common {

class CudaPaddleTensorFactory : public TensorAdapterFactory {
public:
  CudaPaddleTensorFactory() = default;

  virtual ~CudaPaddleTensorFactory() = default;

  std::shared_ptr<TensorAdapter<int64_t>>
  create_int64_t(const std::vector<size_t> &shape) override;

  std::shared_ptr<TensorAdapter<uint8_t>>
  create_uint8_t(const std::vector<size_t> &shape) override;

  std::shared_ptr<TensorAdapter<int64_t>> create_int64_t() override;

  CudaPaddleTensorFactory(const paddle::platform::DeviceContext *device_ctx)
      : _device_ctx(device_ctx) {}

  const paddle::platform::DeviceContext *device_ctx() const {
    return _device_ctx;
  }

private:
  const paddle::platform::DeviceContext *_device_ctx;
};

} // namespace common
