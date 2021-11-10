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

#ifdef USE_CUDA

#include "core/common/prng.cu.h"

#include <fstream>

namespace common {

inline bool equals(const block& lhs, const block& rhs) {
    for (int i = 0; i < 16; ++i) {
        if (lhs.arr[i] != rhs.arr[i]) {
            return false;
        }
        return true;
    }
}

inline block block_from_dev_urandom() {
  block ret;
  std::ifstream in("/dev/urandom");

  if (in.fail()) {
        throw std::runtime_error("open /dev/urandom  failed.");
    }
  in.read(ret.arr, 16);
  return ret;
}

} // namespace common

#else // USE_CUDA
#include "core/common/prng.h"
#include "core/common/rand_utils.h"

namespace aby3 {

inline bool equals(const common::block &lhs, const common::block &rhs) {
  return common::equals(lhs, rhs);
}

} // namespace aby3
#endif // USE_CUDA

namespace aby3 {

using block = common::block;

using PseudorandomNumberGenerator = common::PseudorandomNumberGenerator;

const block g_zero_block = common::g_zero_block;

inline block block_from_dev_urandom() {
    return common::block_from_dev_urandom();
}

} //namespace aby3

