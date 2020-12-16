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

// This op is different with elementwise_add of PaddlePaddle.
// We only consider that the dimensions of X is equal with the dimensions of Y.

#pragma once
#include "mpc_op.h"
#include "paddle/fluid/platform/transform.h"
#include "core/paddlefl_mpc/operators/math/elementwise_op_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MpcElementwiseAddKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override{
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *out_t = ctx.Output<framework::LoDTensor>("Out");

        int axis = ctx.Attr<int>("axis");

        auto out = out_t->mutable_data<T>(ctx.GetPlace());

        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(in_x_t, in_y_t, out_t, axis);
    }
};

template <typename DeviceContext, typename T>
class MpcElementwiseAddGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
        auto *dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
        auto *dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
        int axis = ctx.Attr<int>("axis");

        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add_grad(in_x_t, in_y_t, dout, dx, dy, axis);
    }
};

}  // namespace operators
}  // namespace paddle

