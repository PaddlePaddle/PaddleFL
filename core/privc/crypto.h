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

#include <algorithm>
#include <cstring>
#include <utility>

#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#include "../psi/aes.h"

namespace psi {

typedef unsigned char u8;
typedef unsigned long long u64;
const block ZeroBlock = _mm_set_epi64x(0, 0);
const block OneBlock = _mm_set_epi64x(-1, -1);

static block double_block(block bl);

static inline block hash_block(const block& x, const block& i = ZeroBlock) {
    static AES pi(ZeroBlock);
    block k = double_block(x) ^ i;
    return pi.ecb_enc_block(k) ^ k;
}

static inline std::pair<block, block> hash_blocks(const std::pair<block, block>& x,
                                                  const std::pair<block, block>& i = {ZeroBlock, ZeroBlock}) {
    static AES pi(ZeroBlock);
    block k[2] = {double_block(x.first) ^ i.first, double_block(x.second) ^ i.second};
    block c[2];
    pi.ecb_enc_blocks(k, 2, c);
    return {c[0] ^ k[0], c[1] ^ k[1]};
}

static inline block double_block(block bl) {
    const __m128i mask = _mm_set_epi32(135,1,1,1);
    __m128i tmp = _mm_srai_epi32(bl, 31);
    tmp = _mm_and_si128(tmp, mask);
    tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2,1,0,3));
    bl = _mm_slli_epi32(bl, 1);
    return _mm_xor_si128(bl,tmp);
}

} // namespace psi

