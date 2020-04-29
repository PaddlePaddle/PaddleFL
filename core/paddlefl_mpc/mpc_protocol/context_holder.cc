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

#include "context_holder.h"

namespace paddle {
namespace mpc {

thread_local std::shared_ptr<CircuitContext> ContextHolder::current_mpc_ctx;

thread_local const ExecutionContext *ContextHolder::current_exec_ctx;

thread_local std::shared_ptr<aby3::TensorAdapterFactory>
    ContextHolder::_s_current_tensor_factory;

} // mpc
} // paddle
