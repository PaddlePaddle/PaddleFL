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

const int CURVE_ID = NID_secp160k1;

const int POINT_BUFFER_LEN = 21;
// only apply for 160 bit curve
// specification about point buf len,  see http://www.secg.org/sec1-v2.pdf
// chapter 2.2.3

const int HASH_DIGEST_LEN = SHA_DIGEST_LENGTH;

const int GCM_IV_LEN = 12;

const int GCM_TAG_LEN = 16;

u8 *hash(const void *d, u64 n, void *md);

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

template <typename T>
static inline block to_block(const T& val) {
    block ret = ZeroBlock;
    std::memcpy(&ret, &val, std::min(sizeof ret, sizeof val));
    return ret;
}

// ciphertext = iv || aes_ciphertext || gcm_tag
// allocate buffer before call
int encrypt(const unsigned char *plaintext, int plaintext_len,
            const unsigned char *key, const unsigned char *iv,
            unsigned char *ciphertext);

int decrypt(const unsigned char *ciphertext, int ciphertext_len,
            const unsigned char *key, unsigned char *plaintext);

class ECDH {
private:
    EC_GROUP *_group;
    EC_KEY *_key;
    EC_POINT *_remote_key;
    bool _error;

public:
    ECDH();
    ~ECDH();

    inline bool error() {return _error;}

    ECDH(const ECDH &other) = delete;
    ECDH operator=(const ECDH &other) = delete;

    std::array<u8, POINT_BUFFER_LEN> generate_key();

    std::array<u8, POINT_BUFFER_LEN - 1>
    get_shared_secret(const std::array<u8, POINT_BUFFER_LEN> &remote_key);
};

/*
    This file is part of JustGarble.
    JustGarble is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    JustGarble is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with JustGarble.  If not, see <http://www.gnu.org/licenses/>.
 */


/*------------------------------------------------------------------------
  / OCB Version 3 Reference Code (Optimized C)     Last modified 08-SEP-2012
  /-------------------------------------------------------------------------
  / Copyright (c) 2012 Ted Krovetz.
  /
  / Permission to use, copy, modify, and/or distribute this software for any
  / purpose with or without fee is hereby granted, provided that the above
  / copyright notice and this permission notice appear in all copies.
  /
  / THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
  / WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
  / MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
  / ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  / WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  / ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  / OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  /
  / Phillip Rogaway holds patents relevant to OCB. See the following for
  / his patent grant: http://www.cs.ucdavis.edu/~rogaway/ocb/grant.htm
  /
  / Special thanks to Keegan McAllister for suggesting several good improvements
  /
  / Comments are welcome: Ted Krovetz <ted@krovetz.net> - Dedicated to Laurel K
  /------------------------------------------------------------------------- */

static inline block double_block(block bl) {
    const __m128i mask = _mm_set_epi32(135,1,1,1);
    __m128i tmp = _mm_srai_epi32(bl, 31);
    tmp = _mm_and_si128(tmp, mask);
    tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2,1,0,3));
    bl = _mm_slli_epi32(bl, 1);
    return _mm_xor_si128(bl,tmp);
}

} // namespace psi

