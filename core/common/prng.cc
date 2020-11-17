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

#include "prng.h"

#include <cstring>

namespace common {

PseudorandomNumberGenerator::PseudorandomNumberGenerator(const block &seed)
    : _ctr(0), _now_byte(0) {
  set_seed(seed);
  refill_buffer();
}

void PseudorandomNumberGenerator::set_seed(const block &b) {
  _aes.set_key(b);
  _ctr = 0;

  refill_buffer();
}

void PseudorandomNumberGenerator::refill_buffer() {
  for (auto &ctr : _ctr64) {
    ctr = _mm_cvtsi64_si128(_ctr++);
  }

  _aes.ecb_enc_blocks(_ctr64.data(), _ctr64.size(), _buffer.data());

  _now_byte = 0;
}

void PseudorandomNumberGenerator::get_array(void *res, size_t len) {
  std::array<uint8_t, _s_byte_capacity> &view =
      *reinterpret_cast<std::array<uint8_t, _s_byte_capacity> *>(&_buffer);

  for (size_t write = 0; len > 0;) {
    auto step = std::min(len, _s_byte_capacity - _now_byte);
    std::memcpy(reinterpret_cast<char *>(res) + write, view.data() + _now_byte,
                step);

    write += step;
    _now_byte += step;
    len -= step;
    if (_now_byte == _s_byte_capacity) {
      refill_buffer();
    }
  }
}

} // namespace common
