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

#include "core/privc/crypto.h"
#include "core/privc/privc_context.h"
#include "core/privc/ot.h"

namespace privc {


void ObliviousTransfer::init() {
  auto np_ot_send_pre = [&]() {
    std::array<std::array<std::array<unsigned char,
        psi::POINT_BUFFER_LEN>, 2>, OT_SIZE> send_buffer;

    for (uint64_t idx = 0; idx < OT_SIZE; idx += 1) {
      send_buffer[idx] = _np_ot_sender.send_pre(idx);
    }
    net()->send(next_party(), send_buffer.data(), sizeof(send_buffer));
  };

  auto np_ot_send_post = [&]() {
      std::array<std::array<unsigned char, psi::POINT_BUFFER_LEN>, OT_SIZE> recv_buffer;

      net()->recv(next_party(), recv_buffer.data(), sizeof(recv_buffer));

      for (uint64_t idx = 0; idx < OT_SIZE; idx += 1) {
          _np_ot_sender.send_post(idx, recv_buffer[idx]);
      }
  };

  auto np_ot_recv = [&]() {
      std::array<std::array<std::array<unsigned char,
          psi::POINT_BUFFER_LEN>, 2>, OT_SIZE> recv_buffer;

      std::array<std::array<unsigned char, psi::POINT_BUFFER_LEN>, OT_SIZE> send_buffer;

      net()->recv(next_party(), recv_buffer.data(), sizeof(recv_buffer));

      for (uint64_t idx = 0; idx < OT_SIZE; idx += 1) {
          send_buffer[idx] = _np_ot_recver.recv(idx, recv_buffer[idx]);
      }

      net()->send(next_party(), send_buffer.data(), sizeof(send_buffer));
  };

  _garbled_delta = privc_ctx()->template gen_random_private<block>();
  reinterpret_cast<u8 *>(&_garbled_delta)[0] |= (u8)1;
  _garbled_and_ctr = 0;

  if (party() == 0) {
      np_ot_recv();

      np_ot_send_pre();
      np_ot_send_post();

  } else { // party == Bob
      np_ot_send_pre();
      np_ot_send_post();

      np_ot_recv();
  }
  _ot_ext_sender.init(_base_ot_choices, _np_ot_recver._msgs);
  _ot_ext_recver.init(_np_ot_sender._msgs);
}

} // namespace privc
