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

#include <smmintrin.h>

namespace common {

using block = __m128i;

const block g_zero_block = _mm_set_epi64x(0, 0);

block block_from_dev_urandom();

inline bool equals(const block &lhs, const block &rhs) {
  block neq = _mm_xor_si128(lhs, rhs);
  return _mm_test_all_zeros(neq, neq);
}

} // namespace common
