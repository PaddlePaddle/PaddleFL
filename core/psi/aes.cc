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

#include "aes.h"

#include <wmmintrin.h>

namespace psi {

static block aes128_key_expansion(block key, block key_rcon) {
  key_rcon = _mm_shuffle_epi32(key_rcon, _MM_SHUFFLE(3, 3, 3, 3));
  key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
  key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
  key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
  return _mm_xor_si128(key, key_rcon);
}

AES::AES(const block &user_key) { set_key(user_key); }

void AES::set_key(const block &user_key) {
  _round_key[0] = user_key;
  _round_key[1] = aes128_key_expansion(
      _round_key[0], _mm_aeskeygenassist_si128(_round_key[0], 0x01));
  _round_key[2] = aes128_key_expansion(
      _round_key[1], _mm_aeskeygenassist_si128(_round_key[1], 0x02));
  _round_key[3] = aes128_key_expansion(
      _round_key[2], _mm_aeskeygenassist_si128(_round_key[2], 0x04));
  _round_key[4] = aes128_key_expansion(
      _round_key[3], _mm_aeskeygenassist_si128(_round_key[3], 0x08));
  _round_key[5] = aes128_key_expansion(
      _round_key[4], _mm_aeskeygenassist_si128(_round_key[4], 0x10));
  _round_key[6] = aes128_key_expansion(
      _round_key[5], _mm_aeskeygenassist_si128(_round_key[5], 0x20));
  _round_key[7] = aes128_key_expansion(
      _round_key[6], _mm_aeskeygenassist_si128(_round_key[6], 0x40));
  _round_key[8] = aes128_key_expansion(
      _round_key[7], _mm_aeskeygenassist_si128(_round_key[7], 0x80));
  _round_key[9] = aes128_key_expansion(
      _round_key[8], _mm_aeskeygenassist_si128(_round_key[8], 0x1B));
  _round_key[10] = aes128_key_expansion(
      _round_key[9], _mm_aeskeygenassist_si128(_round_key[9], 0x36));
}

void AES::ecb_enc_block(const block &plaintext, block &cyphertext) const {
  cyphertext = _mm_xor_si128(plaintext, _round_key[0]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[1]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[2]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[3]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[4]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[5]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[6]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[7]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[8]);
  cyphertext = _mm_aesenc_si128(cyphertext, _round_key[9]);
  cyphertext = _mm_aesenclast_si128(cyphertext, _round_key[10]);
}

block AES::ecb_enc_block(const block &plaintext) const {
  block ret;
  ecb_enc_block(plaintext, ret);
  return ret;
}

#define REPEATED_FUNC(func, idx, out, in, k)                                   \
  do {                                                                         \
    out[idx + 0] = func(in[idx + 0], k);                                       \
    out[idx + 1] = func(in[idx + 1], k);                                       \
    out[idx + 2] = func(in[idx + 2], k);                                       \
    out[idx + 3] = func(in[idx + 3], k);                                       \
    out[idx + 4] = func(in[idx + 4], k);                                       \
    out[idx + 5] = func(in[idx + 5], k);                                       \
    out[idx + 6] = func(in[idx + 6], k);                                       \
    out[idx + 7] = func(in[idx + 7], k);                                       \
  } while (0)

void AES::ecb_enc_blocks(const block *plaintexts, size_t block_num,
                         block *cyphertext) const {
  const size_t step = 8;
  size_t idx = 0;
  size_t length = block_num - block_num % step;

  for (; idx < length; idx += step) {
    REPEATED_FUNC(_mm_xor_si128, idx, cyphertext, plaintexts, _round_key[0]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[1]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[2]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[3]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[4]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[5]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[6]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[7]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[8]);
    REPEATED_FUNC(_mm_aesenc_si128, idx, cyphertext, cyphertext, _round_key[9]);
    REPEATED_FUNC(_mm_aesenclast_si128, idx, cyphertext, cyphertext,
                  _round_key[10]);
  }

  for (; idx < block_num; ++idx) {
    cyphertext[idx] = _mm_xor_si128(plaintexts[idx], _round_key[0]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[1]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[2]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[3]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[4]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[5]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[6]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[7]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[8]);
    cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], _round_key[9]);
    cyphertext[idx] = _mm_aesenclast_si128(cyphertext[idx], _round_key[10]);
  }
}
} // namespace psi
