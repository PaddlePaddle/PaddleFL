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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
static inline T GetAttrFromTensor(const framework::Tensor* tensor) {
  const auto* tensor_data = tensor->data<T>();
  framework::Tensor cpu_tensor;
  if (platform::is_gpu_place(tensor->place())) {
    TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<T>();
  }
  return tensor_data[0];
}

template <typename DeviceContext, typename T>
class MpcScaleKernel : public MpcOpKernel<T> {
 public:
  void ComputeImpl(const framework::ExecutionContext& ctx) const override {
    auto* in_var = ctx.InputVar("X");
    auto* in = framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_var);

    T bias = static_cast<T>(ctx.Attr<float>("bias") *
                               std::pow(2, mpc::FIXED_POINTER_SCALING_FACTOR));
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");

    auto scale = ctx.Attr<float>("scale");
    if (ctx.HasInput("ScaleTensor")) {
      auto* scale_tensor = ctx.Input<framework::Tensor>("ScaleTensor");
      scale = GetAttrFromTensor<float>(scale_tensor);
    }

    auto* out_var = ctx.OutputVar("Out");
    if (in_var->IsType<framework::SelectedRows>() && in_var != out_var) {
      auto& in_slr = in_var->Get<framework::SelectedRows>();
      auto* out_slr = out_var->GetMutable<framework::SelectedRows>();
      out_slr->set_rows(in_slr.rows());
      out_slr->set_height(in_slr.height());
    }

    auto* out =
        framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(out_var);
    auto out_ptr = out->mutable_data<T>(in->place());

    PADDLE_ENFORCE_EQ(in->dims(), out->dims(),
                      "in and out should have the same dim");

    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol,
              "Protocol %s is not yet created in MPC Protocol.");
    auto mpc_operator = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
    if (bias_after_scale) {
      mpc_operator->scale(in, scale, out);
      std::for_each(out_ptr, out_ptr + out->numel(), [&bias](T& i) { i += (bias / 3); });
    } else {
      const T* in_data = in->data<T>();
      std::transform(in_data, in_data + in->numel(), out_ptr, [&bias](const T& in){ return in + bias / 3; });
      mpc_operator->scale(in, scale, out);
    }
  }
};

}  // namespace operators
}  // namespace paddle
