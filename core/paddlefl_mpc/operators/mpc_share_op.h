/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

template <typename DeviceContext, typename T>
class MpcShareKernel : public MpcOpKernel<T> {
 public:
  void ComputeImpl(const framework::ExecutionContext& ctx) const override {
    const auto *in_t = ctx.Input<framework::Tensor>("X");
    auto *out_t = ctx.Output<framework::Tensor>("Out");
    auto party = ctx.Attr<int>("party");
    out_t->mutable_data<int64_t>(ctx.GetPlace());

    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol, "Protocol %s is not yet created in MPC Protocol.");
    auto mpc_operator = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
    mpc_operator->online_share(party, in_t,out_t);
  }
};

}  // namespace operators
}  // namespace paddle
