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

#include <memory>
#include <vector>

#include "tensor_adapter.h"

namespace common {

class TensorAdapterFactory {
public:
  TensorAdapterFactory() = default;

  virtual ~TensorAdapterFactory() = default;

  virtual std::shared_ptr<TensorAdapter<int64_t>>
  create_int64_t(const std::vector<size_t> &shape) = 0;

  virtual std::shared_ptr<TensorAdapter<uint8_t>>
  create_uint8_t(const std::vector<size_t> &shape) = 0;

  virtual std::shared_ptr<TensorAdapter<int64_t>> create_int64_t() = 0;

  template <typename T> std::shared_ptr<TensorAdapter<T>> create();

  template <typename T>
  std::shared_ptr<TensorAdapter<T>> create(const std::vector<size_t> &shape);

  template <typename T>
  std::vector<std::shared_ptr<TensorAdapter<T>>> malloc_tensor(size_t size,
                                              const std::vector<size_t>& shape);
};

} // namespace common
