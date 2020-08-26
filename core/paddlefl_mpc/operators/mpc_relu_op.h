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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
//Define forward computation
template <typename DeviceContext, typename T>
class MpcReluKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext& ctx) const override {
        const Tensor* in_t = ctx.Input<Tensor>("X");
<<<<<<< HEAD
        Tensor* out_t = ctx.Output<Tensor>("Out");
=======
        Tensor* out_t = ctx.Output<Tensor>("Y");
>>>>>>> 5a09665c36ffb7eae2288b3f837d3be18091c259
        Tensor* der_t = ctx.Output<Tensor>("Derivative");
        auto x = in_t->data<T>();
        auto y = out_t->mutable_data<T>(ctx.GetPlace());
        auto der = der_t->mutable_data<T>(ctx.GetPlace());
        PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol, "Protocol %s is not yet created in MPC Protocol.");
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()
            ->relu_with_derivative(in_t,out_t, der_t);
  }
};

//Define backward computation
template <typename DeviceContext, typename T>
class MpcReluGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext& ctx) const override {
<<<<<<< HEAD
        auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Out"));
        auto* y_t = ctx.Input<Tensor>("Out");
=======
        auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
        auto* y_t = ctx.Input<Tensor>("Y");
>>>>>>> 5a09665c36ffb7eae2288b3f837d3be18091c259
        auto* der_t = ctx.Input<Tensor>("Derivative");
        auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));
        auto dx = dx_t->mutable_data<T>(ctx.GetPlace());
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->arith_bool_mul(dy_t, der_t, dx_t);
    }
};

}// namespace operaters
}// namespace paddle
