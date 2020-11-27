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

#include <array>
#include <vector>

#include "prng.h"
#include "utils.h"
#include "../privc/common_utils.h"
#include "../privc/typedef.h"

namespace common {

using namespace privc;
//using namespace aby3;
//using TensorBlock = TensorAdapter<int64_t>;
// implementation of ot extension
// generates ot masks
template <typename T> class OTExtBase {
public:
  static const size_t _s_ot_ext_buffer_size = 0x10000;

  // bit size of T
  static const size_t _s_ot_size = sizeof(T) * 8;

  // ot extenstion based on block[128] as 128 * 128 bit matrix
  static const size_t _s_block_bit_size = sizeof(block) * 8;

  // TODO: check if sizeof T is divisible
  static const size_t _s_scale = _s_ot_size / _s_block_bit_size;

  inline static T get_t_from_row(
      const std::array<std::array<block, _s_block_bit_size>, _s_scale> &arr,
      size_t row) {
    T ret;
    auto ptr = reinterpret_cast<block *>(&ret);
    for (size_t i = 0; i < _s_scale; ++i) {
      ptr[i] = arr[i][row];
    }
    return ret;
  }
};

template <typename T> class OTExtSender : public OTExtBase<T> {
  using OTExtBase<T>::_s_ot_ext_buffer_size;
  using OTExtBase<T>::_s_ot_size;
  using OTExtBase<T>::_s_block_bit_size;
  using OTExtBase<T>::_s_scale;

public:
  T _choices;

  void init(const T &_choices, const std::vector<block> &msgs,
            bool init_buffer = false);

  T get_ot_instance();

  void get_ot_instance(TensorBlock* msg);

  template <class U> void fill_ot_buffer(U &send_msg);

private:
  std::array<PseudorandomNumberGenerator, _s_ot_size> _matrix_gen;

  std::array<T, _s_ot_ext_buffer_size> _send_msg;

  void fill_ot_buffer();

  size_t _now_idx;
};

template <typename T> class OTExtReceiver : public OTExtBase<T> {
  using OTExtBase<T>::_s_ot_ext_buffer_size;
  using OTExtBase<T>::_s_ot_size;
  using OTExtBase<T>::_s_block_bit_size;
  using OTExtBase<T>::_s_scale;

public:
  void init(const std::vector<std::array<block, 2>> &msgs,
            bool init_buffer = false);

  std::array<T, 2> get_ot_instance();

  void get_ot_instance(TensorBlock* msg1, TensorBlock* msg2);

  template <class U> void fill_ot_buffer(U &recv_msg);

private:
  std::array<std::array<PseudorandomNumberGenerator, 2>, _s_ot_size>
      _matrix_gen;

  std::array<std::array<T, 2>, _s_ot_ext_buffer_size> _recv_msg;

  void fill_ot_buffer();

  size_t _now_idx;
};
} // namespace common

#include "ot_extension_impl.h"
