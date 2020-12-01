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

#include "./aes.h"

//#include "common_utils.h"
#include "./tensor_adapter.h"

namespace common {

typedef unsigned char u8;
typedef unsigned long long u64;
const block ZeroBlock = _mm_set_epi64x(0, 0);
const block OneBlock = _mm_set_epi64x(-1, -1);
const int POINT_BUFFER_LEN = 21;

using TensorBlock = TensorAdapter<int64_t>;

static block double_block(block bl);

static inline block hash_block(const block& x, const block& i = ZeroBlock) {
    static AES pi(ZeroBlock);
    block k = double_block(x) ^ i;
    return pi.ecb_enc_block(k) ^ k;
}

static inline void hash_block(const TensorBlock* x, TensorBlock* ret,
                              const TensorBlock* i = nullptr) {
    for (int j = 0; j < x->numel() / 2; ++j) {
        block i_block(ZeroBlock);
        if (i) {
            i_block = *(reinterpret_cast<const block*>(i->data()) + j);
        }
        block x_block = *(reinterpret_cast<const block*>(x->data()) + j);
        block* ret_block_ptr = reinterpret_cast<block*>(ret->data()) + j;
        *(ret_block_ptr) = hash_block(x_block, i_block);
    }
}

static inline std::pair<block, block> hash_blocks(const std::pair<block, block>& x,
                                                  const std::pair<block, block>& i = {ZeroBlock, ZeroBlock}) {
    static AES pi(ZeroBlock);
    block k[2] = {double_block(x.first) ^ i.first, double_block(x.second) ^ i.second};
    block c[2];
    pi.ecb_enc_blocks(k, 2, c);
    return {c[0] ^ k[0], c[1] ^ k[1]};
}

static inline void hash_blocks(const std::pair<const TensorBlock*, const TensorBlock*>& x,
                 std::pair<TensorBlock*, TensorBlock*>& ret,
                 const std::pair<TensorBlock*, TensorBlock*>& i = {nullptr, nullptr}) {
    int numel = x.first->numel() / 2;
    for (int j = 0; j < numel; ++j) {
        const block* block_ptr_x_first = reinterpret_cast<const block*>(x.first->data());
        const block* block_ptr_x_second = reinterpret_cast<const block*>(x.second->data());
        block* block_ptr_ret_first = reinterpret_cast<block*>(ret.first->data());
        block* block_ptr_ret_second = reinterpret_cast<block*>(ret.second->data());
        std::pair<block, block> x_pair({*(block_ptr_x_first + j), *(block_ptr_x_second + j)});

        std::pair<block, block> i_pair;
        i_pair.first = i.first ? *(reinterpret_cast<block*>(i.first->data()) + j) : ZeroBlock;
        i_pair.second = i.second ? *(reinterpret_cast<block*>(i.second->data()) + j) : ZeroBlock;

        auto ret_pair = hash_blocks(x_pair, i_pair);
        *(block_ptr_ret_first + j) = ret_pair.first;
        *(block_ptr_ret_second + j) = ret_pair.second;
    }
}

template <typename T>
static inline block to_block(const T& val) {
    block ret = ZeroBlock;
    std::memcpy(&ret, &val, std::min(sizeof ret, sizeof val));
    return ret;
}

template <typename T>
static inline block to_block(const TensorAdapter<T>* val, TensorBlock* ret) {
    block* ret_ptr = reinterpret_cast<block*>(ret->data());
    for (int i = 0; i < val->numel(); ++i) {
        *(ret_ptr + i) = to_block(*(val->data() + i));
    }
}

static inline block double_block(block bl) {
    const __m128i mask = _mm_set_epi32(135,1,1,1);
    __m128i tmp = _mm_srai_epi32(bl, 31);
    tmp = _mm_and_si128(tmp, mask);
    tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2,1,0,3));
    bl = _mm_slli_epi32(bl, 1);
    return _mm_xor_si128(bl,tmp);
}

} // namespace common

