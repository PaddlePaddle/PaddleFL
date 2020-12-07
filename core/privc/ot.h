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

#include <queue>
#include <array>

#include "paddle/fluid/platform/enforce.h"

#include "core/paddlefl_mpc/mpc_protocol/abstract_network.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/common/crypto.h"
#include "core/common/naorpinkas_ot.h"
#include "core/common/ot_extension.h"
#include "utils.h"

namespace privc {

inline std::string block_to_string(const block &b) {
    return std::string(reinterpret_cast<const char *>(&b), sizeof(block));
}

inline u8 block_lsb(const block &val) {
    const u8 *view = reinterpret_cast<const u8 *>(&val);
    return view[0] & (u8)1;
};

inline void block_lsb(const TensorBlock* val, TensorAdapter<u8>* ret) {
    const block* val_ptr = reinterpret_cast<const block*>(val->data());
    std::transform(val_ptr, val_ptr + ret->numel(), ret->data(),
                   [](block a) -> u8 {
                       return block_lsb(a);
                   });
};

inline void block_lsb(const TensorBlock* val, TensorAdapter<int64_t>* ret) {
    const block* val_ptr = reinterpret_cast<const block*>(val->data());
    std::transform(val_ptr, val_ptr + ret->numel(), ret->data(),
                   [](block a) -> int64_t {
                       return (int64_t) block_lsb(a);
                   });
};

inline void gen_ot_masks(OTExtReceiver<block> & ot_ext_recver,
                         uint64_t input,
                         std::vector<block>& ot_masks,
                         std::vector<block>& t0_buffer,
                         size_t word_width = 8 * sizeof(uint64_t)) {
        for (uint64_t idx = 0; idx < word_width; idx += 1) {
            auto ot_instance = ot_ext_recver.get_ot_instance();
            block choice = (input >> idx) & 1 ? common::OneBlock : common::ZeroBlock;

            t0_buffer.emplace_back(ot_instance[0]);
            ot_masks.emplace_back(choice ^ ot_instance[0] ^ ot_instance[1]);
        }
}

inline void gen_ot_masks(OTExtReceiver<block> & ot_ext_recver,
                         const int64_t* input,
                         size_t size,
                         std::vector<block>& ot_masks,
                         std::vector<block>& t0_buffer,
                         size_t word_width = 8 * sizeof(uint64_t)) {
    for (size_t i = 0; i < size; ++i) {
        gen_ot_masks(ot_ext_recver, input[i], ot_masks, t0_buffer, word_width);
    }
}

template <typename T>
inline void gen_ot_masks(OTExtReceiver<block> & ot_ext_recver,
                         const std::vector<T>& input,
                         std::vector<block>& ot_masks,
                         std::vector<block>& t0_buffer,
                         size_t word_width = 8 * sizeof(uint64_t)) {
    for (const auto& i: input) {
        gen_ot_masks(ot_ext_recver, i, ot_masks, t0_buffer, word_width);
    }
}

template <typename T>
inline void gen_ot_masks(OTExtReceiver<block> & ot_ext_recver,
                         const TensorAdapter<T>* input,
                         TensorBlock* ot_masks,
                         TensorBlock* t0_buffer,
                         size_t word_width = 8 * sizeof(uint64_t)) {
    PADDLE_ENFORCE_EQ(ot_masks->numel(),
                    t0_buffer->numel(),
                    "the numel between returns of ot instance0 and ot mask is no match.");
    PADDLE_ENFORCE_EQ(input->numel() * word_width * _g_block_size_expand,
                    t0_buffer->numel(), "the numel of input and returns is no match.");

    auto shape = input->shape();
    auto block_shape = get_block_shape(shape);
    for (uint64_t idx = 0; idx < word_width; idx += 1) {
        auto ot_ins0 = tensor_factory()->template create<int64_t>(block_shape);
        auto ot_ins1 = tensor_factory()->template create<int64_t>(block_shape);
        ot_ext_recver.get_ot_instance(ot_ins0.get(), ot_ins1.get());

        auto choice = tensor_factory()->template create<int64_t>(block_shape);
        block* choice_ptr = reinterpret_cast<block*>(choice->data());
        std::transform(input->data(), input->data() + input->numel(),
                       choice_ptr, [&idx](int64_t a) {
                           return (a >> idx) & 1 ? common::OneBlock : common::ZeroBlock;
                       });

        auto t0_buffer_s = tensor_factory()->template create<int64_t>(block_shape);
        t0_buffer->slice(idx, idx + 1, t0_buffer_s.get());

        ot_ins0->copy(t0_buffer_s.get());

        auto ot_masks_s = tensor_factory()->template create<int64_t>(block_shape);
        ot_masks->slice(idx, idx + 1, ot_masks_s.get());
        ot_masks_s->reshape(block_shape);
        
        choice->bitwise_xor(ot_ins0.get(), ot_masks_s.get());
        ot_masks_s->bitwise_xor(ot_ins1.get(), ot_masks_s.get());
    }
}

class ObliviousTransfer {
public:
  ObliviousTransfer() = delete;
  ObliviousTransfer(block base_ot_choices, block garbled_delta, AbstractNetwork* net, size_t party, size_t next_party) :
        _base_ot_choices(base_ot_choices),
        _net(net),
        _party(party),
        _garbled_delta(garbled_delta),
        _next_party(next_party),
        _np_ot_sender(sizeof(block) * 8),
        _np_ot_recver(sizeof(block) * 8, block_to_string(base_ot_choices)) {
  };

  OTExtReceiver<block>& ot_receiver() { return _ot_ext_recver; }

  OTExtSender<block>& ot_sender() { return _ot_ext_sender; }

  const block& base_ot_choice() const { return _base_ot_choices; }

  const block& garbled_delta() const { return _garbled_delta; }

  u64& garbled_and_ctr() { return _garbled_and_ctr; }

  void base_ot_choice(TensorBlock* ret) const {
    block* ret_ptr = reinterpret_cast<block*>(ret->data());
    std::for_each(ret_ptr, ret_ptr + ret->numel() / _g_block_size_expand,
                    [this](block& a) { a = this->_base_ot_choices; });
  }

  void garbled_delta(TensorBlock* ret) const {
    block* ret_ptr = reinterpret_cast<block*>(ret->data());
    std::for_each(ret_ptr, ret_ptr + ret->numel() / _g_block_size_expand,
                    [this](block& a) { a = this->_garbled_delta; });
  }

  void garbled_and_ctr(TensorAdapter<int64_t>* ret) {
    int64_t* ret_ptr = ret->data();
    std::for_each(ret_ptr, ret_ptr + ret->numel(),
                    [this](int64_t& a) mutable {
                        u64 tmp = ++(this->_garbled_and_ctr);
                        a = *reinterpret_cast<int64_t*>(&tmp);
                    });
  }

  void init();

  static const size_t OT_SIZE = sizeof(block) * 8;

private:

  AbstractNetwork* net() {
      return _net;
  }

  size_t party() {
    return _party;
  }

  size_t next_party() {
    return _next_party;
  }

  const block _base_ot_choices;
  block _garbled_delta;
  u64 _garbled_and_ctr;

  NaorPinkasOTsender _np_ot_sender;
  NaorPinkasOTreceiver _np_ot_recver;

  OTExtSender<block> _ot_ext_sender;
  OTExtReceiver<block> _ot_ext_recver;
  size_t _party;
  size_t _next_party;
  AbstractNetwork* _net;
};

using OT = ObliviousTransfer;

} // namespace privc

