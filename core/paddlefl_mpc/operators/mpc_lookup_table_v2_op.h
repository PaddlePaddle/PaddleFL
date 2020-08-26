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
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/device_context.h"
#include "core/paddlefl_mpc/operators/math/math_function_impl.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T>
class MpcLookupTableV2Kernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &context) const override {
        auto *ids_t = context.Input<Tensor>("Ids");      // int tensor
        auto *output_t = context.Output<Tensor>("Out");  // float tensor
        auto *table_var = context.Input<Tensor>("W");
        auto *ids = ids_t->data<T>();
        auto *table = table_var->data<T>();
        auto *output = output_t->mutable_data<T>(context.GetPlace());

        PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol,
                                "Protocol %s is not yet created in MPC Protocol.");
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->
        mpc_operators()->matmul(ids_t, table_var, output_t);
    }
};

template <typename T>
class MpcLookupTableV2GradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &context) const override {
        auto *ids_t = context.Input<Tensor>("Ids");
        auto id_dim = ids_t->dims();
        auto col_width = id_dim[1];
        auto row_width = id_dim[2];
        auto *d_output_t = context.Input<Tensor>(framework::GradVarName("Out"));
        auto *d_table_t = context.Output<Tensor>(framework::GradVarName("W"));

        // transpose ids_t
        auto *ids = ids_t->data<T>();
        auto *table = d_table_t->mutable_data<T>(context.GetPlace());
        auto *output = d_output_t->data<T>();

        Tensor ids_trans_t;
        auto *ids_trans = ids_trans_t.mutable_data<T>({2, row_width, col_width}, context.GetPlace());

        math::Transpose<platform::CPUDeviceContext, T, 3> transpose;
        auto& dev_ctx = context. template device_context<platform::CPUDeviceContext>();
        transpose(dev_ctx, *ids_t, &ids_trans_t, {0, 2, 1});
        PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol, "Protocol %s is not yet created in MPC Protocol.");
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->matmul(&ids_trans_t, d_output_t, d_table_t);
    }
};

}  // namespace operators
}  // namespace paddle

