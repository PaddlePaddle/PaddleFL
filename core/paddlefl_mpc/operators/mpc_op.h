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

// Description:

#pragma once

#include "paddle/fluid/framework/operator.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_instance.h"
#include "core/privc3/circuit_context.h"

namespace paddle {
namespace operators {

template <typename T> class MpcOpKernel : public framework::OpKernelBase {
public:
  using ELEMENT_TYPE = T;
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_instance()->mpc_protocol(),
                            "Mpc protocol is not yet initialized in executor");

    std::shared_ptr<aby3::CircuitContext> mpc_ctx(
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_context());
    mpc::ContextHolder::template run_with_context<>(&ctx, mpc_ctx,
                                                    [&] { ComputeImpl(ctx); });
  }
  virtual void ComputeImpl(const framework::ExecutionContext &ctx) const = 0;
};

} // namespace operators
} // namespace paddle
