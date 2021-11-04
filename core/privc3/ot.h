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

#pragma once

#include "core/paddlefl_mpc/mpc_protocol/abstract_context.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"

namespace aby3 {

class ObliviousTransfer {
  public:
  template<typename T, template <typename> class Tensor>
  static inline void ot(size_t sender, size_t receiver, size_t helper,
                  const Tensor<T>* choice, const Tensor<T>* m[2],
                  Tensor<T>* buffer[2], Tensor<T>* ret) {
    // TODO: check tensor shape equals
    auto aby3_ctx = paddle::mpc::ContextHolder::mpc_ctx();
    auto tensor_factory = paddle::mpc::ContextHolder::tensor_factory();
    const size_t numel = buffer[0]->numel();
    if (aby3_ctx->party() == sender) {
      bool common = helper == aby3_ctx->next_party();

      auto rand0 = tensor_factory->template create<T>(buffer[0]->shape());
      auto rand1 = tensor_factory->template create<T>(buffer[0]->shape());

      aby3_ctx->template gen_random(*rand0.get(), common);
      aby3_ctx->template gen_random(*rand1.get(), common);

      rand0->bitwise_xor(m[0], buffer[0]);
      rand1->bitwise_xor(m[1], buffer[1]);

      NCCL_GROUP_START
      aby3_ctx->network()->template send(receiver, *buffer[0]);
      aby3_ctx->network()->template send(receiver, *buffer[1]);
      NCCL_GROUP_END

    } else if (aby3_ctx->party() == helper) {
      bool common = sender == aby3_ctx->next_party();

      aby3_ctx->template gen_random(*buffer[0], common);
      aby3_ctx->template gen_random(*buffer[1], common);

      auto tmp0 = tensor_factory->template create<T>(buffer[0]->shape());
      auto tmp1 = tensor_factory->template create<T>(buffer[0]->shape());
      assign_to_tensor(tmp0.get(), (T)1);
      // choice in tmp1
      tmp0->bitwise_and(choice, tmp1.get());
      // tmp0 = r0 ^ r1
      buffer[0]->bitwise_xor(buffer[1], tmp0.get());
      // b1 = choice * (r0 ^ r1)
      tmp0->mul(tmp1.get(), buffer[1]);
      // t0 = choice * (r0 ^ r1) + r0
      buffer[1]->bitwise_xor(buffer[0], tmp0.get());
      aby3_ctx->network()->template send(receiver, *tmp0);

    } else if (aby3_ctx->party() == receiver) {
      NCCL_GROUP_START
      aby3_ctx->network()->template recv(sender, *buffer[0]);
      aby3_ctx->network()->template recv(sender, *buffer[1]);
      aby3_ctx->network()->template recv(helper, *ret);
      NCCL_GROUP_END

      auto tmp0 = tensor_factory->template create<T>(buffer[0]->shape());
      auto tmp1 = tensor_factory->template create<T>(buffer[0]->shape());

      assign_to_tensor(tmp0.get(), (T)1);
      tmp0->bitwise_and(choice, tmp1.get());
      // choice in tmp1
      tmp0->bitwise_and(choice, tmp1.get());
      // tmp0 = r0 ^ r1
      buffer[0]->bitwise_xor(buffer[1], tmp0.get());
      // b1 = choice * (r0 ^ r1)
      tmp0->mul(tmp1.get(), buffer[1]);
      // t0 = choice * (r0 ^ r1) + r0
      buffer[1]->bitwise_xor(buffer[0], tmp0.get());

      ret->bitwise_xor(tmp0.get(), ret);
    }
  }
};

} // namespace aby3
