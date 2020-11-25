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
// a public access context holder, mostly used in the mpc computation where the
// paddle execution
// and mpc circuit context are needed. The corresponding contexts in the
// environment where the operator
// is executed will be stored and accessed here, which are thread local.

#pragma once

#include "paddle/fluid/framework/operator.h"
#include "core/privc3/aby3_context.h"
#include "core/common/paddle_tensor.h"

namespace paddle {
namespace mpc {

using ABY3Context = aby3::ABY3Context;
using ExecutionContext = paddle::framework::ExecutionContext;

class ContextHolder {
public:
  template <typename Operation>
  static void run_with_context(const ExecutionContext *exec_ctx,
                               std::shared_ptr<AbstractContext> mpc_ctx,
                               Operation op) {

    // set new ctxs
    auto old_mpc_ctx = current_mpc_ctx;
    current_mpc_ctx = mpc_ctx;

    auto old_exec_ctx = current_exec_ctx;
    current_exec_ctx = exec_ctx;

    auto old_factory = _s_current_tensor_factory;

    _s_current_tensor_factory = nullptr;

    tensor_factory();

    // run the op
    op();

    // restore ctxs
    current_mpc_ctx = old_mpc_ctx;
    current_exec_ctx = old_exec_ctx;
    _s_current_tensor_factory = old_factory;
  }

  static std::shared_ptr<AbstractContext> mpc_ctx() { return current_mpc_ctx; }

  static const ExecutionContext *exec_ctx() { return current_exec_ctx; }

  static const paddle::platform::DeviceContext *device_ctx() {
    return &current_exec_ctx->device_context();
  }

  static std::shared_ptr<common::TensorAdapterFactory> tensor_factory() {
    if (!_s_current_tensor_factory) {
      _s_current_tensor_factory =
          std::make_shared<common::PaddleTensorFactory>(device_ctx());
    }
    return _s_current_tensor_factory;
  }

private:
  thread_local static std::shared_ptr<AbstractContext> current_mpc_ctx;

  thread_local static const ExecutionContext *current_exec_ctx;

  thread_local static std::shared_ptr<common::TensorAdapterFactory>
      _s_current_tensor_factory;
};

} // mpc
} // paddle
