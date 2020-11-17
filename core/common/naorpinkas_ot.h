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
#include <string>
#include <vector>

#include <emmintrin.h>
#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

namespace common {

const int g_curve_id = NID_secp160k1;
const int g_point_buffer_len = 21;
// only for 160 bit curve
// specification about point buf len,  see http://www.secg.org/sec1-v2.pdf
// chapter 2.2.3

const size_t g_hash_digest_len = SHA_DIGEST_LENGTH;

std::array<uint8_t, g_hash_digest_len> crypto_hash(const void *msg, size_t n);

using block = __m128i;

class NaorPinkasOTsender {
public:
  NaorPinkasOTsender() = delete;

  NaorPinkasOTsender(const NaorPinkasOTsender &other) = delete;

  NaorPinkasOTsender &operator=(const NaorPinkasOTsender &other) = delete;

  NaorPinkasOTsender(size_t ot_size);

  ~NaorPinkasOTsender();

  std::vector<std::array<block, 2>> _msgs;

  std::array<std::array<uint8_t, g_point_buffer_len>, 2>
  send_pre(const size_t idx);

  void send_post(const size_t idx,
                 const std::array<uint8_t, g_point_buffer_len> &input);

private:
  const size_t _ot_size;

  EC_GROUP *_group;

  std::vector<EC_KEY *> _gr; // if you find this naming is rigid, plz refer to
  std::vector<EC_KEY *> _c;  // https://dblp.org/rec/conf/soda/NaorP01

  std::vector<EC_POINT *> _cr;
  std::vector<EC_POINT *> _pk0r;
};

class NaorPinkasOTreceiver {
public:
  NaorPinkasOTreceiver() = delete;

  NaorPinkasOTreceiver(const NaorPinkasOTreceiver &other) = delete;

  NaorPinkasOTreceiver &operator=(const NaorPinkasOTreceiver &other) = delete;

  NaorPinkasOTreceiver(size_t ot_size, const std::string &choices);

  ~NaorPinkasOTreceiver();

  std::vector<block> _msgs;

  std::array<uint8_t, g_point_buffer_len>
  recv(const size_t idx,
       const std::array<std::array<uint8_t, g_point_buffer_len>, 2> &input);

private:
  const size_t _ot_size;
  const std::string _choices;
  EC_GROUP *_group;
  std::vector<EC_KEY *> _k_sigma;
  std::vector<EC_POINT *> _pk0;
  std::vector<EC_POINT *> _c;
  std::vector<EC_POINT *> _gr;
};
} // namespace common
