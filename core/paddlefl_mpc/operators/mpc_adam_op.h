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

#include <math.h>

#include "./math/math_function.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators.h"

namespace paddle {
namespace operators {

static inline float GetAttrFromTensor(const framework::Tensor* tensor) {
  const float* tensor_data = tensor->data<float>();
  framework::Tensor cpu_tensor;
  return tensor_data[0];
}

template <typename DeviceContext, typename T, typename T1>
class MpcAdamOpKernel : public MpcOpKernel<T> {
  public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override{
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE(param_var->IsType<framework::LoDTensor>(),
                   "The Var(%s)'s type should be LoDTensor, "
                   "but the received is %s",
                   ctx.InputNames("Param").front(),
                   framework::ToTypeName(param_var->Type()));

    using paddle::framework::LoDTensor;

    T1 epsilon = static_cast<T1>(ctx.Attr<float>("epsilon"));
    auto* param = ctx.Input<LoDTensor>("Param");
    auto* grad_var = ctx.InputVar("Grad");
    auto* mom1 = ctx.Input<LoDTensor>("Moment1");
    auto* mom2 = ctx.Input<LoDTensor>("Moment2");
    auto* lr = ctx.Input<LoDTensor>("LearningRate");

    auto* beta1_pow = ctx.Input<LoDTensor>("Beta1Pow");
    auto* beta2_pow = ctx.Input<LoDTensor>("Beta2Pow");

    auto* param_out = ctx.Output<LoDTensor>("ParamOut");
    auto* mom1_out = ctx.Output<LoDTensor>("Moment1Out");
    auto* mom2_out = ctx.Output<LoDTensor>("Moment2Out");
    auto* beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");

    T1 beta1 = static_cast<T1>(ctx.Attr<float>("beta1"));
    if (ctx.HasInput("Beta1Tensor")) {
      auto* beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      PADDLE_ENFORCE_EQ(beta1_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta1Tensor) size must be 1, but get %d",
                            beta1_tensor->numel()));
      beta1 = static_cast<T1>(GetAttrFromTensor(beta1_tensor));
    }
    T1 beta2 = static_cast<T1>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      PADDLE_ENFORCE_EQ(beta2_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(Beta2Tensor) size must be 1, but get %d",
                            beta2_tensor->numel()));
      beta2 = static_cast<T1>(GetAttrFromTensor(beta2_tensor));
    }
    VLOG(3) << "beta1_pow.numel() : " << beta1_pow->numel()
            << "beta2_pow.numel() : " << beta2_pow->numel();
    VLOG(3) << "param.numel(): " << param->numel();

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto* grad = ctx.Input<LoDTensor>("Grad");

      // AdamFunctor<T, CPUAdam> functor(
      //     beta1, beta2, epsilon, beta1_pow->data<T>(), beta2_pow->data<T>(),
      //     mom1->data<T>(), mom1_out->mutable_data<T>(ctx.GetPlace()),
      //     mom2->data<T>(), mom2_out->mutable_data<T>(ctx.GetPlace()),
      //     lr->data<T>(), grad->data<T>(), param->data<T>(),
      //     param_out->mutable_data<T>(ctx.GetPlace()));
      // functor(param->numel());

      T1 lr_value = *lr->template data<T1>();

      T1 beta1_pow_ = *beta1_pow->template data<T1>();
      T1 beta2_pow_ = *beta2_pow->template data<T1>();

      double lr_ =  lr_value * sqrt(1 - beta2_pow_) / (1 - beta1_pow_);

      framework::Tensor temp;
      temp.mutable_data<T>(param->dims(), ctx.GetPlace());

      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(grad, (1 - beta1), &temp);
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(mom1, beta1, mom1_out);
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(mom1_out, &temp, mom1_out);

      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(grad, (1 - beta2), &temp);
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(grad, &temp, &temp);
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(mom2, beta2, mom2_out);
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(mom2_out, &temp, mom2_out);

      // mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(grad, lr[0], &temp);
      // mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->sub(param, &temp, param_out);

      math::SetConstant<DeviceContext, T> set_const;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      set_const(
          dev_ctx,
          &temp,
          T(epsilon * pow(2, mpc::ABY3_SCALING_FACTOR) / 3));

      // temp = epsilon + mom2_out
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(mom2_out, &temp, &temp);
      // temp = 1 / sqrt(epsilon + mom2_out)
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->inverse_square_root(&temp, &temp);
      // temp = mom1_out / sqrt(epsilon + mom2_out)
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(mom1_out, &temp, &temp);
      // temp = lr * mom1_out / sqrt(epsilon + mom2_out)
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(&temp, lr_, &temp);
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->sub(param, &temp, param_out);

      beta1_pow_out->mutable_data<T1>(ctx.GetPlace())[0] =
          beta1 * beta1_pow->template data<T1>()[0];
      beta2_pow_out->mutable_data<T1>(ctx.GetPlace())[0] =
          beta2 * beta2_pow->template data<T1>()[0];

    } else {
      PADDLE_THROW("Variable type not supported by adam_op");
    }

    }
};
}  // namespace operators
}  // namespace paddle
