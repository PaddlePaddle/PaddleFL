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

#include <emmintrin.h>

namespace psi {

using block = __m128i;

class AES {
public:
  AES() {}

  AES(const block &user_key);

  AES(const AES &other) = delete;

  AES &operator=(const AES &other) = delete;

  void set_key(const block &user_key);

  void ecb_enc_block(const block &plaintext, block &cyphertext) const;

  block ecb_enc_block(const block &plaintext) const;

  void ecb_enc_blocks(const block *plaintexts, size_t block_num,
                      block *ciphertext) const;

private:
  block _round_key[11];
};

} // namespace psi
