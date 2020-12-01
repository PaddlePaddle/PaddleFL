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

#include "ot_extension.h"

#include <stdexcept>

#include "sse_transpose.h"

namespace common {

template <typename T>
void OTExtSender<T>::init(const T &choices, const std::vector<block> &msgs,
                          bool init_buffer) {

  if (msgs.size() != _s_ot_size) {
    throw std::invalid_argument("ot ext error: "
                                "num of msgs mismatched for choices");
  }

  _choices = choices;

  size_t idx = 0;
  for (auto &gen : _matrix_gen) {
    gen.set_seed(msgs[idx++]);
  }

  if (init_buffer) {
    fill_ot_buffer();
  } else {
    _now_idx = _s_ot_ext_buffer_size;
  }
}

template <typename T>
void OTExtReceiver<T>::init(const std::vector<std::array<block, 2>> &msgs,
                            bool init_buffer) {
  if (msgs.size() != _s_ot_size) {
    throw std::invalid_argument("ot ext error: "
                                "num of msgs mismatched for choices");
  }

  size_t idx = 0;
  for (auto &gen : _matrix_gen) {
    gen[0].set_seed(msgs[idx][0]);
    gen[1].set_seed(msgs[idx++][1]);
  }

  if (init_buffer) {
    fill_ot_buffer();
  } else {
    _now_idx = _s_ot_ext_buffer_size;
  }
}

template <typename T>
template <class U>
void OTExtSender<T>::fill_ot_buffer(U &send_msg) {
  std::array<std::array<block, _s_block_bit_size>, _s_scale> q;

  std::array<block, _s_ot_size> &q_block_view =
      *reinterpret_cast<std::array<block, _s_ot_size> *>(&q);

  // 128 OT instances in one batch
  for (size_t msg_idx = 0; msg_idx < send_msg.size();) {
    size_t gen_idx = 0;
    for (auto &slot : q_block_view) {
      slot = _matrix_gen[gen_idx++].template get<block>();
    }

    for (auto &sub_mat : q) {
      sse_transpose128(sub_mat);
    }

    for (size_t row_idx = 0;
         row_idx < _s_block_bit_size && msg_idx < send_msg.size();
         ++row_idx, ++msg_idx) {

      send_msg[msg_idx] = this->get_t_from_row(q, row_idx);
    }
  }
}

template <typename T> void OTExtSender<T>::fill_ot_buffer() {
  fill_ot_buffer(_send_msg);
  _now_idx = 0;
}

template <typename T>
template <class U>
void OTExtReceiver<T>::fill_ot_buffer(U &recv_msg) {
  std::array<std::array<block, _s_block_bit_size>, _s_scale> t[2];

  std::array<block, _s_ot_size> &t0_block_view =
      *reinterpret_cast<std::array<block, _s_ot_size> *>(&t[0]);

  std::array<block, _s_ot_size> &t1_block_view =
      *reinterpret_cast<std::array<block, _s_ot_size> *>(&t[1]);

  // 128 OT instances in one batch
  for (size_t msg_idx = 0; msg_idx < recv_msg.size();) {
    for (size_t gen_idx = 0; gen_idx < _s_ot_size; ++gen_idx) {
      t0_block_view[gen_idx] = _matrix_gen[gen_idx][0].template get<block>();
      t1_block_view[gen_idx] = _matrix_gen[gen_idx][1].template get<block>();
    }

    for (auto &sub_mat : t[0]) {
      sse_transpose128(sub_mat);
    }

    for (auto &sub_mat : t[1]) {
      sse_transpose128(sub_mat);
    }

    for (size_t row_idx = 0; row_idx < 128 && msg_idx < recv_msg.size();
         ++row_idx, ++msg_idx) {
      recv_msg[msg_idx][0] = this->get_t_from_row(t[0], row_idx);
      recv_msg[msg_idx][1] = this->get_t_from_row(t[1], row_idx);
    }
  }
}

template <typename T> void OTExtReceiver<T>::fill_ot_buffer() {
  fill_ot_buffer(_recv_msg);
  _now_idx = 0;
}

template <typename T> T OTExtSender<T>::get_ot_instance() {
  if (_now_idx == _s_ot_ext_buffer_size) {
    fill_ot_buffer();
  }

  return _send_msg[_now_idx++];
}

template <typename T> std::array<T, 2> OTExtReceiver<T>::get_ot_instance() {
  if (_now_idx == _s_ot_ext_buffer_size) {
    fill_ot_buffer();
  }

  return _recv_msg[_now_idx++];
}
} // namespace common
