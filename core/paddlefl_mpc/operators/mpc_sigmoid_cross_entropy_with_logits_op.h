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

#pragma once
#include "mpc_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// Out = sigmoid(x) : prediction of x. 
// todo: Out = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename DeviceContext, typename T>
class MpcSigmoidCrossEntropyWithLogitsKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *in_x_t = ctx.Input<Tensor>("X");
        auto *out_t = ctx.Output<Tensor>("Out");
        out_t->mutable_data<T>(ctx.GetPlace());

        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->sigmoid(in_x_t, out_t);
    }
};

// dX = sigmoid(X) - labels
template <typename DeviceContext, typename T>
class MpcSigmoidCrossEntropyWithLogitsGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *in_label_t = ctx.Input<Tensor>("Label");
        auto *in_sigmoid_t = ctx.Input<Tensor>("Out");
        auto dx = ctx.Output<Tensor>(framework::GradVarName("X"));

        auto dx_data = dx->mutable_data<T>(ctx.GetPlace());

        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->sub(in_sigmoid_t, in_label_t, dx);
    }
};

}  // namespace operators
}  // namespace paddle
