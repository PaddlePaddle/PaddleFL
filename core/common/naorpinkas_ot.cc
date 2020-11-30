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

#include "naorpinkas_ot.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include <openssl/err.h>
#include <openssl/sha.h>

namespace common {

inline void throw_openssl_error() {
  throw std::runtime_error("openssl error: " + std::to_string(ERR_get_error()));
}

std::array<uint8_t, g_hash_digest_len> crypto_hash(const void *msg, size_t n) {
  std::array<uint8_t, g_hash_digest_len> md;
  SHA1(reinterpret_cast<const uint8_t *>(msg), n, md.data());
  return md;
}

NaorPinkasOTsender::NaorPinkasOTsender(size_t ot_size) : _ot_size(ot_size) {

  _msgs.resize(_ot_size);

  int ret = 0;

  _group = EC_GROUP_new_by_curve_name(g_curve_id);
  if (_group == NULL) {
    throw_openssl_error();
  }

  for (size_t idx = 0; idx < _ot_size; ++idx) {
    auto c = EC_KEY_new();
    if (_group == NULL) {
      throw_openssl_error();
    }

    ret = EC_KEY_set_group(c, _group);
    if (ret != 1) {
      throw_openssl_error();
    }

    auto gr = EC_KEY_new();
    if (gr == NULL) {
      throw_openssl_error();
    }

    ret = EC_KEY_set_group(gr, _group);
    if (ret != 1) {
      throw_openssl_error();
    }

    auto cr = EC_POINT_new(_group);
    if (cr == NULL) {
      throw_openssl_error();
    }

    auto pk0r = EC_POINT_new(_group);
    if (pk0r == NULL) {
      throw_openssl_error();
    }
    _c.emplace_back(c);
    _gr.emplace_back(gr);
    _pk0r.emplace_back(pk0r);
    _cr.emplace_back(cr);
  }
}

NaorPinkasOTsender::~NaorPinkasOTsender() {
  for (auto &item : _cr) {
    EC_POINT_free(item);
  }
  for (auto &item : _pk0r) {
    EC_POINT_free(item);
  }
  for (auto &item : _gr) {
    EC_KEY_free(item);
  }
  for (auto &item : _c) {
    EC_KEY_free(item);
  }
  EC_GROUP_free(_group);
}

std::array<std::array<uint8_t, g_point_buffer_len>, 2>
NaorPinkasOTsender::send_pre(const size_t idx) {
  std::array<std::array<uint8_t, g_point_buffer_len>, 2> output;

  int ret = EC_KEY_generate_key(_c[idx]);
  if (ret != 1) {
    throw_openssl_error();
  }

  const EC_POINT *C_point = EC_KEY_get0_public_key(_c[idx]);
  if (C_point == NULL) {
    throw_openssl_error();
  }

  ret = EC_POINT_point2oct(_group, C_point, POINT_CONVERSION_COMPRESSED,
                           output[0].data(), g_point_buffer_len, NULL);
  if (ret == 0) {
    throw_openssl_error();
  }

  ret = EC_KEY_generate_key(_gr[idx]);
  if (ret != 1) {
    throw_openssl_error();
  }

  const EC_POINT *gr_point = EC_KEY_get0_public_key(_gr[idx]);
  if (C_point == NULL) {
    throw_openssl_error();
  }

  ret = EC_POINT_point2oct(_group, gr_point, POINT_CONVERSION_COMPRESSED,
                           output[1].data(), g_point_buffer_len, NULL);
  if (ret == 0) {
    throw_openssl_error();
  }

  return output;
}

void NaorPinkasOTsender::send_post(
    const size_t idx, const std::array<uint8_t, g_point_buffer_len> &input) {
  int ret = 0;

  const BIGNUM *r = EC_KEY_get0_private_key(_gr[idx]);
  if (r == NULL) {
    throw_openssl_error();
  }

  const EC_POINT *C_point = EC_KEY_get0_public_key(_c[idx]);
  if (C_point == NULL) {
    throw_openssl_error();
  }

  ret = EC_POINT_mul(_group, _cr[idx], NULL, C_point, r, NULL);
  if (ret != 1) {
    throw_openssl_error();
  }

  ret =
      EC_POINT_oct2point(_group, _pk0r[idx], input.data(), input.size(), NULL);
  if (ret != 1) {
    throw_openssl_error();
  }

  ret = EC_POINT_mul(_group, _pk0r[idx], NULL, _pk0r[idx], r, NULL);
  if (ret != 1) {
    throw_openssl_error();
  }

  uint8_t msg0[g_point_buffer_len];
  uint8_t msg1[g_point_buffer_len];

  ret = EC_POINT_point2oct(_group, _pk0r[idx], POINT_CONVERSION_COMPRESSED,
                           msg0, g_point_buffer_len, NULL);
  if (ret == 0) {
    throw_openssl_error();
  }

  ret = EC_POINT_invert(_group, _pk0r[idx], NULL);
  if (ret != 1) {
    throw_openssl_error();
  }

  // pk0r = c^r - pk0^r
  ret = EC_POINT_add(_group, _pk0r[idx], _cr[idx], _pk0r[idx], NULL);
  if (ret != 1) {
    throw_openssl_error();
  }

  ret = EC_POINT_point2oct(_group, _pk0r[idx], POINT_CONVERSION_COMPRESSED,
                           msg1, g_point_buffer_len, NULL);
  if (ret == 0) {
    throw_openssl_error();
  }

  // sigma also need to be added to the hash accroding to the paper
  msg0[0] = 0;
  msg1[0] = 1;

  auto md = crypto_hash(msg0, g_point_buffer_len);
  std::memcpy(&_msgs[idx][0], md.data(), sizeof(block));

  md = crypto_hash(msg1, g_point_buffer_len);
  std::memcpy(&_msgs[idx][1], md.data(), sizeof(block));
}

NaorPinkasOTreceiver::NaorPinkasOTreceiver(size_t ot_size,
                                           const std::string &choices)
    : _ot_size(ot_size), _choices(choices) {

  if (choices.size() * 8 < ot_size) {
    throw std::invalid_argument("np ot error: choices too short for ot_size");
  }

  _msgs.resize(_ot_size);

  int ret = 0;

  _group = EC_GROUP_new_by_curve_name(g_curve_id);
  if (_group == NULL) {
    throw_openssl_error();
  }

  for (size_t idx = 0; idx < _ot_size; ++idx) {

    auto k_sigma = EC_KEY_new();
    if (k_sigma == NULL) {
      throw_openssl_error();
    }

    ret = EC_KEY_set_group(k_sigma, _group);
    if (ret != 1) {
      throw_openssl_error();
    }

    auto pk0 = EC_POINT_new(_group);
    if (pk0 == NULL) {
      throw_openssl_error();
    }

    auto c = EC_POINT_new(_group);
    if (c == NULL) {
      throw_openssl_error();
    }

    auto gr = EC_POINT_new(_group);
    if (gr == NULL) {
      throw_openssl_error();
    }
    _k_sigma.emplace_back(k_sigma);
    _pk0.emplace_back(pk0);
    _c.emplace_back(c);
    _gr.emplace_back(gr);
  }
}

NaorPinkasOTreceiver::~NaorPinkasOTreceiver() {
  for (auto &item : _gr) {
    EC_POINT_free(item);
  }
  for (auto &item : _c) {
    EC_POINT_free(item);
  }
  for (auto &item : _pk0) {
    EC_POINT_free(item);
  }
  for (auto &item : _k_sigma) {
    EC_KEY_free(item);
  }
  EC_GROUP_free(_group);
}

std::array<uint8_t, g_point_buffer_len> NaorPinkasOTreceiver::recv(
    const size_t idx,
    const std::array<std::array<uint8_t, g_point_buffer_len>, 2> &input) {

  int ret = 0;
  std::array<uint8_t, g_point_buffer_len> out_put_pk0;

  auto choices_bit_view = [this, idx]() {
    const uint8_t *bit_view =
        reinterpret_cast<const uint8_t *>(_choices.data());
    uint8_t ret = bit_view[idx / 8] >> (idx % 8);
    return ret & 1;
  };

  if (idx >= 8 * _choices.size()) {
    throw std::invalid_argument("np ot error: choices idx exceed, idx = " +
                                std::to_string(idx) + " / " +
                                std::to_string(_choices.size() * 8));
  }

  uint8_t sigma = choices_bit_view();

  ret = EC_KEY_generate_key(_k_sigma[idx]);
  if (ret != 1) {
    throw_openssl_error();
  }

  ret = EC_POINT_copy(_pk0[idx], EC_KEY_get0_public_key(_k_sigma[idx]));
  if (ret != 1) {
    throw_openssl_error();
  }

  ret = EC_POINT_oct2point(_group, _c[idx], input[0].data(), input[0].size(),
                           NULL);
  if (ret != 1) {
    throw_openssl_error();
  }

  ret = EC_POINT_oct2point(_group, _gr[idx], input[1].data(), input[1].size(),
                           NULL);

  if (ret != 1) {
    throw_openssl_error();
  }

  if (sigma) {
    ret = EC_POINT_invert(_group, _pk0[idx], NULL);
    if (ret != 1) {
      throw_openssl_error();
    }

    ret = EC_POINT_add(_group, _pk0[idx], _c[idx], _pk0[idx], NULL);
    if (ret != 1) {
      throw_openssl_error();
    }
  }

  ret = EC_POINT_point2oct(_group, _pk0[idx], POINT_CONVERSION_COMPRESSED,
                           out_put_pk0.data(), g_point_buffer_len, NULL);
  if (ret == 0) {
    throw_openssl_error();
  }

  // pk0 = pk_sigma_r
  ret = EC_POINT_mul(_group, _pk0[idx], NULL, _gr[idx],
                     EC_KEY_get0_private_key(_k_sigma[idx]), NULL);
  if (ret != 1) {
    throw_openssl_error();
  }

  uint8_t msg[g_point_buffer_len];
  ret = EC_POINT_point2oct(_group, _pk0[idx], POINT_CONVERSION_COMPRESSED, msg,
                           g_point_buffer_len, NULL);
  if (ret == 0) {
    throw_openssl_error();
  }

  msg[0] = sigma;

  auto md = crypto_hash(msg, g_point_buffer_len);
  std::memcpy(&_msgs[idx], md.data(), sizeof(block));

  return out_put_pk0;
}
} // namespace common
