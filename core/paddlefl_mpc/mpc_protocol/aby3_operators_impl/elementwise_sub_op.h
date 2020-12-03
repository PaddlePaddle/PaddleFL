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

// Description: implementations of elementwise_sub_op according to ABY3 protocol
#pragma once

#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/common.h"

namespace paddle {
namespace operators {
namespace aby3impl {

using namespace paddle::operators::math;
using namespace paddle::mpc;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;

static void sub_impl(const Tensor *lhs, const Tensor *rhs, Tensor *out) {
    auto lhs_tuple = from_tensor(lhs);
    auto rhs_tuple = from_tensor(rhs);
    auto out_tuple = from_tensor(out);

    auto lhs_ = std::get<0>(lhs_tuple).get();
    auto rhs_ = std::get<0>(rhs_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    lhs_->sub(rhs_, out_);
}

} // aby3impl
} // operators
} // paddle

