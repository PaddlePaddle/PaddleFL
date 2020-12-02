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

// Description: implementations of each virtual op according to ABY3 protocol

#pragma once

#include <utility>

#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_operators.h"
#include "paddle/fluid/framework/tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/privc3/aby3_context.h"
#include "core/privc3/fixedpoint_tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/common/paddle_tensor.h"

namespace paddle {
namespace operators {
namespace aby3 {

using paddle::framework::Tensor;

void add_impl(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1);

void add_grad_impl(const Tensor *in_x_t, const Tensor *in_y_t, const Tensor *dout, Tensor *dx, Tensor *dy, int axis = -1);

} // aby3
} // operators
} // paddle
