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

template <typename DeviceContext, typename T>
class MpcSumKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto in_vars = ctx.MultiInputVar("X");
        size_t in_num = in_vars.size();
        auto out_var = ctx.OutputVar("Out");
        bool in_place = out_var == in_vars[0];

        if (out_var->IsType<framework::LoDTensor>()) {
            auto *out = out_var->GetMutable<framework::LoDTensor>();
            auto *out_ptr = out->mutable_data<T>(ctx.GetPlace());
            if (in_num >= 1 && in_vars[0]->IsType<framework::LoDTensor>()) {
                auto &in_0_tensor = in_vars[0]->Get<framework::LoDTensor>();
                if (in_0_tensor.numel() > 0) {
                    in_place = (in_0_tensor.data<T>() == out_ptr);
                }
             }
             int start = in_place ? 1 : 0;
             if (!in_place) {
                 if ((in_num >= 2) && in_vars[0]->IsType<framework::LoDTensor>() &&
                         in_vars[1]->IsType<framework::LoDTensor>()) {
                     auto &in_0 = in_vars[0]->Get<framework::LoDTensor>();
                     auto &in_1 = in_vars[1]->Get<framework::LoDTensor>();
                     if (in_0.numel() && in_1.numel()) {
                         mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(&in_0, &in_1, out);
                         start = 2;
                     }
                 }
                 if (start != 2) {
                     auto t = framework::EigenVector<T>::Flatten(*out);
                     auto &device_ctx = ctx.template device_context<DeviceContext>();
                     t.device(*device_ctx.eigen_device()) = t.constant(static_cast<T>(0));
                 }
             }

             // If in_place, just skip the first tensor
             for (size_t i = start; i < in_num; i++) {
                 if (in_vars[i]->IsType<framework::LoDTensor>()) {
                     auto &in_t = in_vars[i]->Get<framework::LoDTensor>();
                     if (in_t.numel() == 0) {
                         continue;
                     }
                     mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(out, &in_t, out);
                 } else {
                     PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
                 }
             }
        }else {
            PADDLE_THROW("Unexpected branch, output variable type is %s",
                         framework::ToTypeName(out_var->Type()));
        }
    }
};
}  // namespace operators
}  // namespace paddle

