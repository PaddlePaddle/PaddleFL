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
#include <vector>

namespace aby3 {

template <typename T> class TensorAdapter {
public:
  TensorAdapter() = default;

  virtual ~TensorAdapter() = default;

  virtual T *data() = 0;

  virtual const T *data() const = 0;

  virtual std::vector<size_t> shape() const = 0;

  virtual void reshape(const std::vector<size_t> &shape) = 0;

  virtual size_t numel() const = 0;

  virtual void copy(TensorAdapter *ret) const {
    // TODO: check shape equals
    std::copy(data(), data() + numel(), ret->data());
  }

  // element wise op, need operands' dim are same
  virtual void add(const TensorAdapter *rhs, TensorAdapter *ret) const = 0;

  // element wise op, need operands' dim are same
  virtual void sub(const TensorAdapter *rhs, TensorAdapter *ret) const = 0;

  virtual void negative(TensorAdapter *ret) const = 0;

  // element wise op, need operands' dim are same
  virtual void mul(const TensorAdapter *rhs, TensorAdapter *ret) const = 0;

  // element wise op, need operands' dim are same
  virtual void div(const TensorAdapter *rhs, TensorAdapter *ret) const = 0;

  // 2d matrix muliply,  need operands' rank are 2
  virtual void mat_mul(const TensorAdapter *rhs, TensorAdapter *ret) const = 0;

  // element wise op, need operands' dim are same
  virtual void bitwise_xor(const TensorAdapter *rhs,
                           TensorAdapter *ret) const = 0;

  // element wise op, need operands' dim are same
  virtual void bitwise_and(const TensorAdapter *rhs,
                           TensorAdapter *ret) const = 0;

  // element wise op, need operands' dim are same
  virtual void bitwise_or(const TensorAdapter *rhs,
                          TensorAdapter *ret) const = 0;

  // element wise op, need operands' dim are same
  virtual void bitwise_not(TensorAdapter *ret) const = 0;

  virtual void lshift(size_t rhs, TensorAdapter *ret) const = 0;

  virtual void rshift(size_t rhs, TensorAdapter *ret) const = 0;

  virtual void logical_rshift(size_t rhs, TensorAdapter *ret) const = 0;

  // when using an integer type T as fixed-point number
  // value of T val is interpreted as val / 2 ^ scaling_factor()
  virtual size_t scaling_factor() const = 0;

  virtual size_t &scaling_factor() = 0;

  // slice by shape[0]
  // e.g. x.shape = [ 2, 3, 4]
  //      x.slice(1, 2, y)
  //      y.shape = [ 1, 3, 4]
  virtual void slice(size_t begin_idx, size_t end_idx,
                     TensorAdapter *out) const = 0;
};
} // namespace aby3
