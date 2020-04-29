// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "mpc_op.h"
#include "paddle_encrypted/mpc_protocol/mpc_instance.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
// Define forward computation
template <typename DeviceContext, typename T>
class MpcReluKernel : public MpcOpKernel<T> {
public:
  void ComputeImpl(const framework::ExecutionContext &ctx) const override {
    const Tensor *in_t = ctx.Input<Tensor>("X");
    Tensor *out_t = ctx.Output<Tensor>("Y");
    auto x = in_t->data<T>();
    auto y = out_t->mutable_data<T>(ctx.GetPlace());
    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol,
                            "Protocol %s is not yet created in MPC Protocol.");
    mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->relu(
        in_t, out_t);
  }
};

// Define backward computation
template <typename DeviceContext, typename T>
class MpcReluGradKernel : public MpcOpKernel<T> {
public:
  void ComputeImpl(const framework::ExecutionContext &ctx) const override {
    auto *dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *y_t = ctx.Input<Tensor>("Y");
    auto *dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());
    mpc::MpcInstance::mpc_instance()
        ->mpc_protocol()
        ->mpc_operators()
        ->relu_grad(y_t, dy_t, dx_t, 0.0);
  }
};

} // namespace operaters
} // namespace paddle
