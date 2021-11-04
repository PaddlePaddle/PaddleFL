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

#ifdef USE_AES_NI
#include <wmmintrin.h>
#endif

namespace common {

#ifdef USE_AES_NI
static block aes128_key_expansion(block key, block key_rcon) {
    key_rcon = _mm_shuffle_epi32(key_rcon, _MM_SHUFFLE(3, 3, 3, 3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, key_rcon);
}

void AES::set_key(const block& user_key) {
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

void AES::ecb_enc_block(const block& plaintext, block& cyphertext) const {
    block cipher = _mm_xor_si128(plaintext, _round_key[0]);
    cipher = _mm_xor_si128(plaintext, _round_key[0]);
    cipher = _mm_aesenc_si128(cipher, _round_key[1]);
    cipher = _mm_aesenc_si128(cipher, _round_key[2]);
    cipher = _mm_aesenc_si128(cipher, _round_key[3]);
    cipher = _mm_aesenc_si128(cipher, _round_key[4]);
    cipher = _mm_aesenc_si128(cipher, _round_key[5]);
    cipher = _mm_aesenc_si128(cipher, _round_key[6]);
    cipher = _mm_aesenc_si128(cipher, _round_key[7]);
    cipher = _mm_aesenc_si128(cipher, _round_key[8]);
    cipher = _mm_aesenc_si128(cipher, _round_key[9]);
    cipher = _mm_aesenclast_si128(cipher, _round_key[10]);
    cyphertext = cipher;
}

#else
// openssl aes
void AES::set_key(const block& user_key) {
    // sizeof block = 128 bit
    AES_set_encrypt_key(reinterpret_cast<const unsigned char*>(&user_key),
                        128, &_aes_key);
}

void AES::ecb_enc_block(const block& plaintext, block& cyphertext) const {
    AES_encrypt(reinterpret_cast<const unsigned char*>(&plaintext),
                reinterpret_cast<unsigned char*>(&cyphertext),
                &_aes_key);
}
#endif

void AES::ecb_enc_blocks(const block* plaintexts, size_t block_num,
                         block* cyphertext) const {

#pragma omp parallel num_threads(4)
#pragma omp for
    for (size_t i = 0; i < block_num; ++i) {
        ecb_enc_block(plaintexts[i], cyphertext[i]);
    }
}

AES::AES(const block& user_key) { set_key(user_key); }

block AES::ecb_enc_block(const block& plaintext) const {
    block ret;
    ecb_enc_block(plaintext, ret);
    return ret;
}

} // namespace common
