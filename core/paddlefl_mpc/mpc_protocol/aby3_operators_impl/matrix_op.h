/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Description: implementations of matrix ops(mul) according to ABY3 protocol

#pragma once

#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {
namespace aby3 {

using paddle::framework::Tensor;

void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
                bool trans_lhs = false, bool trans_rhs = false);

void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, int x_num_col_dims, int y_num_col_dims);

void mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *out, Tensor *dx, Tensor *dy, int x_num_col_dims, int y_num_col_dims);

} // aby3
} // operators
} // paddle
