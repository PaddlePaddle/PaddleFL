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
#include <cmath>

#include <emmintrin.h>

namespace psi {

using block = __m128i;

using Block512 = std::array<block, 4>;

inline Block512 operator^(const Block512 &lhs, const Block512 &rhs) {
  Block512 ret;

  ret[0] = lhs[0] ^ rhs[0];
  ret[1] = lhs[1] ^ rhs[1];
  ret[2] = lhs[2] ^ rhs[2];
  ret[3] = lhs[3] ^ rhs[3];

  return ret;
}

inline Block512 operator&(const Block512 &lhs, const Block512 &rhs) {
  Block512 ret;

  ret[0] = lhs[0] & rhs[0];
  ret[1] = lhs[1] & rhs[1];
  ret[2] = lhs[2] & rhs[2];
  ret[3] = lhs[3] & rhs[3];

  return ret;
}

// the maximum size of the stash for Cuckoo hashing.
inline size_t get_stash_size(size_t input) {

  if (input >= (1 << 24)) {
    return 2;
  }
  if (input >= (1 << 20)) {
    return 3;
  }
  if (input >= (1 << 16)) {
    return 4;
  }
  if (input >= (1 << 12)) {
    return 6;
  }
  if (input >= (1 << 8)) {
    return 12;
  }
  return 12; // other cases
}

// length of the psuedorandom code (and hence the width of the OT
// extension matrix) in BaRK-OPRF protocol.
inline size_t get_codeword_size(size_t input) {
  if (input >= (1 << 24)) {
    return 448 / 8; // in byte
  }
  if (input >= (1 << 20)) {
    return 448 / 8;
  }
  if (input >= (1 << 16)) {
    return 440 / 8;
  }
  if (input >= (1 << 12)) {
    return 432 / 8;
  }
  if (input >= (1 << 8)) {
    return 424 / 8;
  }
  return 424 / 8;
}

// length of the BaRK-OPRF output messages.
inline size_t get_mask_size(size_t input, size_t other,
                            size_t stat_sec_param = 40) {
  return (stat_sec_param + std::log2(input * other) + 7) / 8;
}

} // namespace psi
