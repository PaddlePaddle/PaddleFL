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

// Description:
// abstract mpc operation interface

#pragma once

#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace mpc {

using paddle::framework::Tensor;

class MpcOperators {
public:
  virtual void add(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void neg(const Tensor *op, Tensor *out) = 0;

  virtual void sum(const Tensor *op, Tensor *out) = 0;

  virtual void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void scale(const Tensor *lhs, const double factor, Tensor *out) = 0;

  virtual void relu(const Tensor *op, Tensor *out) = 0;

  virtual void softmax(const Tensor *op, Tensor *out) = 0;

  virtual void gt(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void geq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void lt(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void leq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void eq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void neq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

  virtual void relu_grad(const Tensor *y, const Tensor *dy, Tensor *dx,
                         const float point) = 0;
};

} // mpc
} // paddle
