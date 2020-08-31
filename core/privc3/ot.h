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
    const size_t numel = buffer[0]->numel();
    if (aby3_ctx->party() == sender) {
      bool common = helper == aby3_ctx->next_party();
      aby3_ctx->template gen_random(*buffer[0], common);
      aby3_ctx->template gen_random(*buffer[1], common);
      for (size_t i = 0; i < numel; ++i) {
        buffer[0]->data()[i] ^= m[0]->data()[i];
        buffer[1]->data()[i] ^= m[1]->data()[i];
      }
      aby3_ctx->network()->template send(receiver, *buffer[0]);
      aby3_ctx->network()->template send(receiver, *buffer[1]);

    } else if (aby3_ctx->party() == helper) {
      bool common = sender == aby3_ctx->next_party();

      aby3_ctx->template gen_random(*buffer[0], common);
      aby3_ctx->template gen_random(*buffer[1], common);

      for (size_t i = 0; i < numel; ++i) {
        buffer[0]->data()[i] = choice->data()[i] & 1 ?
            buffer[1]->data()[i] : buffer[0]->data()[i];
      }
      aby3_ctx->network()->template send(receiver, *buffer[0]);
    } else if (aby3_ctx->party() == receiver) {
      aby3_ctx->network()->template recv(sender, *buffer[0]);
      aby3_ctx->network()->template recv(sender, *buffer[1]);
      aby3_ctx->network()->template recv(helper, *ret);
      size_t i = 0;
      std::for_each(ret->data(), ret->data() + numel, [&buffer, &i, choice, ret](T& in) {
                    bool c = choice->data()[i] & 1;
                    in ^= buffer[c]->data()[i];
                    ++i;}
                    );
    }
  }
};

} // namespace aby3