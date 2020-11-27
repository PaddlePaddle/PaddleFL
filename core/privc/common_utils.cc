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

#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/privc/privc_context.h"
#include "core/privc3/tensor_adapter_factory.h"
#include "../privc3/tensor_adapter.h"
#include "typedef.h"
#include "ot.h"
#include "common_utils.h"

namespace privc {

static std::shared_ptr<AbstractContext> privc_ctx() {
  return paddle::mpc::ContextHolder::mpc_ctx();
}

static std::shared_ptr<OT> ot() {
    return std::dynamic_pointer_cast<PrivCContext>(privc_ctx())->ot();
}

static std::shared_ptr<aby3::TensorAdapterFactory> tensor_factory() {
    return paddle::mpc::ContextHolder::tensor_factory();
}

static std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> tripletor() {
    return std::dynamic_pointer_cast<PrivCContext>(privc_ctx())->triplet_generator();
}

static size_t party() {
    return privc_ctx()->party();
}

static size_t next_party() {
    return privc_ctx()->next_party();
}

static AbstractNetwork* net() {
  return privc_ctx()->network();
}

} // namespace privc