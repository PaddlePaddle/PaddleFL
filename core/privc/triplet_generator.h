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
#include "core/common/prng.h"
#include "core/common/crypto.h"
#include "core/common/naorpinkas_ot.h"
#include "core/common/ot_extension.h"
#include "core/common/tensor_adapter.h"
//#include "core/privc3/prng_utils.h"
//#include "core/privc/crypto.h"
//#include "core/psi/naorpinkas_ot.h"
//#include "core/psi/ot_extension.h"
//#include "core/privc3/tensor_adapter.h"
#include "core/privc/ot.h"
#include "core/privc/privc_context.h"

namespace privc {

using AbstractNetwork = paddle::mpc::AbstractNetwork;
using AbstractContext = paddle::mpc::AbstractContext;
using block = common::block;
using NaorPinkasOTsender = common::NaorPinkasOTsender;
using NaorPinkasOTreceiver = common::NaorPinkasOTreceiver;

template<typename T>
using OTExtSender = common::OTExtSender<T>;
template<typename T>
using OTExtReceiver = common::OTExtReceiver<T>;

template <typename T>
using TensorAdapter = common::TensorAdapter<T>;

template<size_t N>
inline int64_t fixed64_mult(const int64_t a, const int64_t b) {

  __int128_t res = (__int128_t)a * (__int128_t)b;

  return static_cast<int64_t>(res >> N);
}

template<size_t N>
inline int64_t signedfixed128_mult(int64_t a, int64_t b) {
    int res[4];
    *reinterpret_cast<__int128_t *>(res) = (__int128_t)a * (__int128_t)b;
    return *reinterpret_cast<int64_t *>(&res[1]);
}

template<size_t N>
inline uint64_t lshift(uint64_t lhs, size_t rhs) {
    return signedfixed128_mult<N>(lhs, (uint64_t)1 << rhs);
}

inline std::string block_to_string(const block &b) {
    return std::string(reinterpret_cast<const char *>(&b), sizeof(block));
}

template<typename T, size_t N>
class TripletGenerator {
public:
  TripletGenerator(common::PseudorandomNumberGenerator* prng,
                   block base_ot_choices, AbstractNetwork* net,
                   size_t party, size_t next_party) :
        //_base_ot_choices(circuit_context->gen_random_private<block>()),
        _prng(prng),
        _base_ot_choices(base_ot_choices),
        _np_ot_sender(sizeof(block) * 8),
        _np_ot_recver(sizeof(block) * 8, block_to_string(_base_ot_choices)),
        _net(net),
        _party(party),
        _next_party(next_party) {};

  void init();

  virtual void get_triplet(TensorAdapter<T>* ret);

  // TODO: use SecureML sec4.2 triplet generator trick to improve mat_mul
  virtual void get_penta_triplet(TensorAdapter<T>* ret);

  std::queue<std::array<T, 3>> _triplet_buffer;
  std::queue<std::array<T, 5>> _penta_triplet_buffer;

  static const size_t _s_triplet_step = 1 << 8;
  static constexpr double _s_fixed_point_compensation = 0.3;

protected:
  // dummy type for specilize template method
  template<typename T_>
  class Type2Type {
    typedef T_ type;
  };

  void fill_triplet_buffer() { fill_triplet_buffer_impl<T>(Type2Type<T>()); }

  template<typename T__>
  void fill_triplet_buffer_impl(const Type2Type<T__>) {
    PADDLE_THROW("type except `int64_t` for generating triplet is not implemented yet");
  }

  // specialize template method by overload
  template<typename T__>
  void fill_triplet_buffer_impl(const Type2Type<int64_t>);

  void fill_penta_triplet_buffer() { fill_penta_triplet_buffer_impl<T>(Type2Type<T>()); }

  template<typename T__>
  void fill_penta_triplet_buffer_impl(const Type2Type<T__>) {
    PADDLE_THROW("type except `int64_t` for generating triplet is not implemented yet");
  }

  // specialize template method by overload
  template<typename T__>
  void fill_penta_triplet_buffer_impl(const Type2Type<int64_t>);

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
  std::shared_ptr<OT> ot() {
    return std::dynamic_pointer_cast<PrivCContext>(privc_ctx())->ot();
  }
  // gen triplet for int64_t type
  std::vector<uint64_t> gen_product(const std::vector<uint64_t> &input);
  std::vector<std::pair<uint64_t, uint64_t>> gen_product(size_t ot_sender,
                                                 const std::vector<uint64_t> &input0,
                                                 const std::vector<uint64_t> &input1
                                                 = std::vector<uint64_t>());

  template<typename U> U gen_random_private() { return _prng->get<U>(); }

  const block _base_ot_choices;

  NaorPinkasOTsender _np_ot_sender;
  NaorPinkasOTreceiver _np_ot_recver;

  OTExtSender<block> _ot_ext_sender;
  OTExtReceiver<block> _ot_ext_recver;
  //std::shared_ptr<AbstractContext> _privc_ctx;
  common::PseudorandomNumberGenerator* _prng;
  AbstractNetwork* _net;
  size_t _party;
  size_t _next_party;
};

} // namespace privc

#include "triplet_generator_impl.h"
