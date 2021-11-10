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
#include "paddle/fluid/framework/eigen.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct GetLearningRate{
    double operator()(const framework::Tensor* t);
};

template <typename DeviceContext, typename T, typename T1>
class MpcSGDOpKernel : public MpcOpKernel<T> {
  public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override{
        const auto *param_var = ctx.InputVar("Param");
        PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                    "The Var(%s)'s type should be LoDTensor, "
                    "but the received is %s",
                    ctx.InputNames("Param").front(),
                    framework::ToTypeName(param_var->Type()));

        const auto *grad_var = ctx.InputVar("Grad");
        PADDLE_ENFORCE_EQ(grad_var->IsType<framework::LoDTensor>(), true,
                    "The Var(%s)'s type should be LoDTensor, "
                    "but the received is %s",
                    ctx.InputNames("Grad").front(),
                    framework::ToTypeName(grad_var->Type()));

        const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");
        const auto *param = ctx.Input<framework::Tensor>("Param");
        const auto *grad = ctx.Input<framework::Tensor>("Grad");

        auto *param_out = ctx.Output<framework::Tensor>("ParamOut");

        auto sz = param_out->numel();
        PADDLE_ENFORCE_EQ(param->numel(), sz);
        PADDLE_ENFORCE_EQ(grad->numel(), sz);

        auto get_lr_functor = GetLearningRate<DeviceContext, T1>();

        double lr = get_lr_functor(learning_rate);

        param_out->mutable_data<T>(ctx.GetPlace());
        PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol, "Protocol %s is not yet created in MPC Protocol.");
        // update parameters
        framework::Tensor temp;
        temp.mutable_data<T>(param->dims(), ctx.GetPlace());
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(grad, lr, &temp);
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->sub(param, &temp, param_out);
    }
};
}  // namespace operators
}  // namespace paddle
