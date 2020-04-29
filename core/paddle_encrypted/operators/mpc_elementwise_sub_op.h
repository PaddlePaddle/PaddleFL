// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

// This op is different with elementwise_sub of PaddlePaddle.
// We only consider that the dimensions of X is equal with the dimensions of Y.

#pragma once
#include "mpc_op.h"
#include "paddle_encrypted/mpc_protocol/mpc_instance.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MpcElementwiseSubKernel : public MpcOpKernel<T> {
public:
  void ComputeImpl(const framework::ExecutionContext &ctx) const override {
    auto *in_x_t = ctx.Input<Tensor>("X");
    auto *in_y_t = ctx.Input<Tensor>("Y");
    auto *out_t = ctx.Output<Tensor>("Out");

    auto out = out_t->mutable_data<T>(ctx.GetPlace());
    mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->sub(
        in_x_t, in_y_t, out_t);
  }
};

template <typename DeviceContext, typename T>
class MpcElementwiseSubGradKernel : public MpcOpKernel<T> {
public:
  void ComputeImpl(const framework::ExecutionContext &ctx) const override {
    VLOG(3) << "******** MpcElementwiseSubGradKernel: ";
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto dout_data = dout->data<T>();

    if (dx) {
      auto dx_data = dx->mutable_data<T>(ctx.GetPlace());
      for (size_t i = 0; i < dout->numel(); i++) {
        dx_data[i] = dout_data[i];
      }
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->neg(
          dout, dy);
    }
  }
};

} // namespace operators
} // namespace paddle
