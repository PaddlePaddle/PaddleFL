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
#include "core/privc3/prng_utils.h"
#include "core/privc/crypto.h"
#include "core/psi/naorpinkas_ot.h"
#include "core/psi/ot_extension.h"

namespace privc {

using AbstractNetwork = paddle::mpc::AbstractNetwork;
using AbstractContext = paddle::mpc::AbstractContext;
using block = psi::block;
using NaorPinkasOTsender = psi::NaorPinkasOTsender;
using NaorPinkasOTreceiver = psi::NaorPinkasOTreceiver;
using u64 = psi::u64;
using u8 = psi::u8;

template<typename T>
using OTExtSender = psi::OTExtSender<T>;
template<typename T>
using OTExtReceiver = psi::OTExtReceiver<T>;


inline std::string block_to_string(const block &b) {
    return std::string(reinterpret_cast<const char *>(&b), sizeof(block));
}

inline u8 block_lsb(const block &val) {
    const u8 *view = reinterpret_cast<const u8 *>(&val);
    return view[0] & (u8)1;
};

inline void gen_ot_masks(OTExtReceiver<block> & ot_ext_recver,
                         uint64_t input,
                         std::vector<block>& ot_masks,
                         std::vector<block>& t0_buffer,
                         size_t word_width = 8 * sizeof(uint64_t)) {
        for (uint64_t idx = 0; idx < word_width; idx += 1) {
            auto ot_instance = ot_ext_recver.get_ot_instance();
            block choice = (input >> idx) & 1 ? psi::OneBlock : psi::ZeroBlock;

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

class ObliviousTransfer {
public:
  ObliviousTransfer(std::shared_ptr<AbstractContext>& circuit_context) :
        _base_ot_choices(circuit_context->gen_random_private<block>()),
        _np_ot_sender(sizeof(block) * 8),
        _np_ot_recver(sizeof(block) * 8, block_to_string(_base_ot_choices)) {
      _privc_ctx = circuit_context;
  };

  OTExtReceiver<block>& ot_receiver() { return _ot_ext_recver; }

  OTExtSender<block>& ot_sender() { return _ot_ext_sender; }

  const block& base_ot_choice() const { return _base_ot_choices; }

  const block& garbled_delta() const { return _garbled_delta; }

  u64& garbled_and_ctr() { return _garbled_and_ctr; }

  void init();

  static const size_t OT_SIZE = sizeof(block) * 8;

private:

  std::shared_ptr<AbstractContext> privc_ctx() {
      return _privc_ctx;
  }
  AbstractNetwork* net() {
      return _privc_ctx->network();
  }

  size_t party() {
    return privc_ctx()->party();
  }

  size_t next_party() {
    return privc_ctx()->next_party();
  }

  const block _base_ot_choices;
  block _garbled_delta;
  u64 _garbled_and_ctr;

  NaorPinkasOTsender _np_ot_sender;
  NaorPinkasOTreceiver _np_ot_recver;

  OTExtSender<block> _ot_ext_sender;
  OTExtReceiver<block> _ot_ext_recver;
  std::shared_ptr<AbstractContext> _privc_ctx;
};

using OT = ObliviousTransfer;
} // namespace privc

