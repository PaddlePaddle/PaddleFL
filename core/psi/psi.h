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
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuckoo_hash.h"
#include "../common/naorpinkas_ot.h"
#include "../common/ot_extension.h"
#include "../common/utils.h"

namespace psi {

using common::Block512;
using common::block;
using common::NaorPinkasOTreceiver;
using common::NaorPinkasOTsender;
using common::PseudorandomNumberGenerator;
using common::OTExtReceiver;
using common::OTExtSender;
using common::AES;

class PsiBase {
public:
  PsiBase(size_t sender_size, size_t recver_size, const block &seed);

  PsiBase(const PsiBase &other) = delete;

  PsiBase &operator=(const PsiBase &other) = delete;

  virtual ~PsiBase() {}

  size_t oprf_output_len() const { return _oprf_output_len; }

  size_t sender_size() const { return _sender_size; }

  size_t code_word_width() const { return _code_word_width; }

  size_t cuckoo_bins_num() const { return _bin_num; }

  void init_input(const std::set<std::string> &input);

  PseudorandomNumberGenerator &prng() { return _prng; }

protected:
  const size_t _sender_size;

  const size_t _recver_size;

  const size_t _bin_num;

  const size_t _max_stash_size;

  const size_t _code_word_width;

  const size_t _oprf_output_len;

  PseudorandomNumberGenerator _prng;

  std::vector<std::string> _input;

  std::array<std::vector<block>, 4> _aes_hash_tab;

  std::vector<size_t> _intersection;
};

class PsiSender : public PsiBase {
public:
  PsiSender(size_t sender_size, size_t recver_size, const block &seed);

  virtual ~PsiSender();

  PsiSender(const PsiSender &other) = delete;

  PsiSender &operator=(const PsiSender &other) = delete;

  void init_offline(const std::set<std::string> &input);

  void sync();

  void recv_masks(size_t begin_idx, size_t end_idx,
                  const std::vector<Block512> &masks);

  const std::vector<uint8_t> &send_oprf_outputs(size_t idx);

  NaorPinkasOTreceiver &np_ot() { return _np_ot; }

private:
  void init_collector();

private:
  const Block512 _ot_ext_choices;

  OTExtSender<Block512> _ot_ext;

  std::vector<Block512> _ot_sender_msgs;

  SimpleHasher _bins;

  NaorPinkasOTreceiver _np_ot;

  std::vector<uint8_t> _output_buf[3];

  std::array<std::vector<size_t>, 3> _permute_table;

  size_t _permute_now_idx[3];
};

class PsiReceiver : public PsiBase {
public:
  PsiReceiver(size_t sender_size, size_t recver_size, const block &seed);

  virtual ~PsiReceiver() {}

  PsiReceiver(const PsiReceiver &other) = delete;

  PsiReceiver &operator=(const PsiReceiver &other) = delete;

  inline size_t stash_bins_num() { return _bins._stash.size(); }

  void init_offline(const std::set<std::string> &input);

  void sync();

  std::vector<Block512> send_masks(size_t begin_idx, size_t end_idx);

  void recv_oprf_outputs(size_t idx, const std::vector<std::string> &input);

  std::vector<std::string> output();

  NaorPinkasOTsender &np_ot() { return _np_ot; }

private:
  void init_collector();

private:
  std::vector<std::array<Block512, 2>> _ot_recver_msgs;

  OTExtReceiver<Block512> _ot_ext;

  CuckooHasher _bins;

  NaorPinkasOTsender _np_ot;

  // use 3 map to keep local result of bins
  // first 64 bit of hash used as key, if key found, check if val also matched
  std::unordered_map<uint64_t, std::pair<block, uint64_t>> _bin_result[3];

  std::vector<std::pair<block, uint64_t>> _stash_result;
};
} // namespace psi
