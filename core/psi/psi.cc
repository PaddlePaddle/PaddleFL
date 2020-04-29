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

#include "psi.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "aes.h"

namespace psi {

PsiBase::PsiBase(size_t sender_size, size_t recver_size, const block &seed)
    : _sender_size(sender_size), _recver_size(recver_size),
      _bin_num(1.2 * recver_size), _max_stash_size(get_stash_size(recver_size)),
      _code_word_width(get_codeword_size(sender_size)),
      _oprf_output_len(get_mask_size(sender_size, recver_size)), _prng(seed) {
  if (_oprf_output_len > sizeof(block)) {
    throw std::invalid_argument("psi error: oprf output length exceed");
  }
}

void encode_input(std::array<std::vector<block>, 4> &aes_hash_tab,
                  const std::vector<block> &hashed_input) {
  AES aes_cipher[4];
  for (size_t key = 0; key < 4; ++key) {
    aes_cipher[key].set_key(_mm_set1_epi64x(key));
  }
  for (auto &a : aes_hash_tab) {
    a.clear();
    a.resize(hashed_input.size());
  }
  aes_cipher[0].ecb_enc_blocks(hashed_input.data(), hashed_input.size(),
                               aes_hash_tab[0].data());
  aes_cipher[1].ecb_enc_blocks(hashed_input.data(), hashed_input.size(),
                               aes_hash_tab[1].data());
  aes_cipher[2].ecb_enc_blocks(hashed_input.data(), hashed_input.size(),
                               aes_hash_tab[2].data());
  aes_cipher[3].ecb_enc_blocks(hashed_input.data(), hashed_input.size(),
                               aes_hash_tab[3].data());
}

void init_input(std::vector<std::string> &output,
                std::array<std::vector<block>, 4> &aes_hash_tab,
                const std::set<std::string> &input) {
  for (auto &x : input) {
    output.emplace_back(x);
  }
  std::vector<block> hashed_input;
  for (auto &item : output) {
    auto md = crypto_hash(item.data(), item.size());
    // block size == 128 bit
    block &md_block = *reinterpret_cast<block *>(&md);
    hashed_input.emplace_back(md_block);
  }
  encode_input(aes_hash_tab, hashed_input);
}

void PsiBase::init_input(const std::set<std::string> &input) {
  _input.clear();
  return ::psi::init_input(_input, _aes_hash_tab, input);
}

inline std::string block512_to_string(const Block512 &b) {
  return std::string(reinterpret_cast<const char *>(&b), sizeof(b));
}

PsiSender::PsiSender(size_t sender_size, size_t recver_size, const block &seed)
    : PsiBase(sender_size, recver_size, seed),
      _ot_ext_choices(prng().template get<Block512>()), _ot_ext(),
      // cuckoo size is decided by recver_size
      _bins(recver_size), _np_ot(512, block512_to_string(_ot_ext_choices)),
      _permute_now_idx{} {
  for (auto &buf : _output_buf) {
    buf.resize(_sender_size * _oprf_output_len);
  }
  // incase of init list failed
  // _np_ot._choices = block512_to_string(_ot_ext_choices);
}
PsiSender::~PsiSender() {}

void PsiSender::init_collector() {
  for (auto &p : _permute_table) {
    p.resize(_sender_size);
    size_t idx = 0;
    for (auto &val : p) {
      val = idx++;
    }
    std::shuffle(p.begin(), p.end(), prng());
  }
  for (auto &val : _permute_now_idx) {
    val = 0;
  }
}

void PsiSender::init_offline(const std::set<std::string> &input) {
  init_input(input);
  _bins.insert_all(_aes_hash_tab);
  init_collector();
}

void PsiSender::sync() {
  _ot_ext.init(_ot_ext_choices, _np_ot._msgs);
  _ot_sender_msgs.resize(_bin_num + _max_stash_size);
  _ot_ext.fill_ot_buffer(_ot_sender_msgs);
}

void PsiSender::recv_masks(size_t begin_idx, size_t end_idx,
                           const std::vector<Block512> &masks) {
  if (masks.size() != (end_idx - begin_idx)) {
    throw std::invalid_argument("psi error: mask num mismatched");
  }
  for (size_t bin_idx = begin_idx; bin_idx < end_idx; ++bin_idx) {
    auto get_oprf_output_lambda = [this, &masks, bin_idx, begin_idx](
        size_t item_idx, size_t hash_idx) {

      Block512 code_word;

      code_word[0] = _aes_hash_tab[0][item_idx];
      code_word[1] = _aes_hash_tab[1][item_idx];
      code_word[2] = _aes_hash_tab[2][item_idx];
      code_word[3] = _aes_hash_tab[3][item_idx];

      auto oprf_input =
          _ot_sender_msgs[bin_idx] ^
          ((masks[bin_idx - begin_idx] ^ code_word) & _ot_ext_choices);

      auto md = crypto_hash(oprf_input.data(), _code_word_width);

      auto &now_idx = _permute_now_idx[hash_idx];

      std::memcpy(_output_buf[hash_idx].data() +
                      _permute_table[hash_idx][now_idx++] * _oprf_output_len,
                  md.data(), _oprf_output_len);
    };

    if (bin_idx < _bin_num) {
      for (auto &bin_item : _bins._table[bin_idx]) {
        get_oprf_output_lambda(bin_item.item_idx, bin_item.hash_idx);
      }
    } else if (bin_idx < _bin_num + _max_stash_size) {
      // outputbuf[0] and permute[0] reused
      std::shuffle(_permute_table[0].begin(), _permute_table[0].end(), prng());
      _permute_now_idx[0] = 0;
      for (size_t idx = 0; idx < _sender_size; ++idx) {
        get_oprf_output_lambda(idx, 0);
      }
    } else {
      throw std::runtime_error("psi error: bin idx exceed");
    }
  }
}

const std::vector<uint8_t> &PsiSender::send_oprf_outputs(size_t idx) {
  if (idx >= _max_stash_size) {
    throw std::invalid_argument("psi error: idx exceed");
  }

  // buf[0] reused for stash
  return _output_buf[(idx < 3 ? idx : 0)];
}

PsiReceiver::PsiReceiver(size_t sender_size, size_t recver_size,
                         const block &seed)
    : PsiBase(sender_size, recver_size, seed), _ot_ext(), _bins(recver_size),
      _np_ot(512) {}

void PsiReceiver::init_collector() {
  for (auto &r : _bin_result) {
    r.reserve(_bins._bins.size());
  }
  _stash_result.resize(_bins._stash.size());
}

void PsiReceiver::init_offline(const std::set<std::string> &input) {
  init_input(input);
  _bins.insert_all(_aes_hash_tab);
  if (_bins._stash.size() > _max_stash_size) {
    throw std::runtime_error("psi error: stash size exceed");
  }
  init_collector();
}

void PsiReceiver::sync() {
  _ot_ext.init(_np_ot._msgs);
  _ot_recver_msgs.resize(_bin_num + _max_stash_size);
  _ot_ext.fill_ot_buffer(_ot_recver_msgs);
}

std::vector<Block512> PsiReceiver::send_masks(size_t begin_idx,
                                              size_t end_idx) {
  std::vector<Block512> ret_val(end_idx - begin_idx);
  size_t ret_idx = 0;
  for (size_t bin_idx = begin_idx; bin_idx < end_idx; ++bin_idx) {
    Bin bin_item;
    bool bin_item_flag = false;

    if (bin_idx < _bin_num) {
      bin_item = _bins._bins[bin_idx];
      bin_item_flag = true;

    } else if (bin_idx < _bin_num + _bins._stash.size()) {
      bin_item = _bins._stash[bin_idx - _bin_num];

    } else {
      throw std::runtime_error("psi error: bin_idx exceed");
    }
    Block512 code_word;
    Block512 mask_to_send;

    if (bin_item.is_empty() == false) {

      code_word[0] = _aes_hash_tab[0][bin_item.item_idx];
      code_word[1] = _aes_hash_tab[1][bin_item.item_idx];
      code_word[2] = _aes_hash_tab[2][bin_item.item_idx];
      code_word[3] = _aes_hash_tab[3][bin_item.item_idx];

      mask_to_send =
          code_word ^ _ot_recver_msgs[bin_idx][0] ^ _ot_recver_msgs[bin_idx][1];

      auto md =
          crypto_hash(_ot_recver_msgs[bin_idx][0].data(), _code_word_width);

      std::memset(md.data() + _oprf_output_len, 0,
                  md.size() - _oprf_output_len);

      if (bin_item_flag == true) {
        _bin_result[bin_item.hash_idx].emplace(
            *reinterpret_cast<uint64_t *>(md.data()),
            std::pair<block, uint64_t>(*reinterpret_cast<block *>(md.data()),
                                       bin_item.item_idx));
      } else {
        _stash_result[bin_idx - _bin_num] = std::pair<block, uint64_t>(
            *reinterpret_cast<block *>(md.data()), bin_item.item_idx);
      }
    } else {
      mask_to_send = prng().template get<Block512>();
    }
    ret_val[ret_idx++] = mask_to_send;
  }
  return ret_val;
}
void PsiReceiver::recv_oprf_outputs(size_t hash_idx,
                                    const std::vector<std::string> &input) {
  if (hash_idx >= 3 + _max_stash_size) {
    // 3 hashes used in cuckoo hash
    throw std::invalid_argument("psi error: input hash idx mismatched");
  }
  if (hash_idx < 3) {
    for (size_t buf_idx = 0; buf_idx < input.size(); ++buf_idx) {
      uint64_t key[2] = {0, 0};

      std::memcpy(key, input[buf_idx].data(), _oprf_output_len);
      auto match = _bin_result[hash_idx].find(key[0]);

      if (match != _bin_result[hash_idx].end() &&
          (_oprf_output_len <= sizeof(uint64_t) ||
           key[1] == reinterpret_cast<uint64_t *>(
                         // cmp high bits
                         &match->second.first)[1])) {
        _intersection.emplace_back(match->second.second);
      }
    }
  } else if (hash_idx < 3 + _bins._stash.size()) {
    std::unordered_map<uint64_t, uint64_t> buf_items;
    buf_items.reserve(input.size());

    for (size_t buf_idx = 0; buf_idx < input.size(); ++buf_idx) {
      uint64_t key[2] = {0, 0};

      std::memcpy(key, input[buf_idx].data(), _oprf_output_len);

      buf_items.emplace(std::pair<uint64_t, uint64_t>(key[0], key[1]));
    }

    uint64_t our_item_val[2];
    uint64_t our_item_idx = _stash_result[hash_idx - 3].second;

    std::memcpy(our_item_val, &_stash_result[hash_idx - 3].first,
                sizeof(block));

    auto match = buf_items.find(our_item_val[0]);

    if (match != buf_items.end() && (_oprf_output_len <= sizeof(uint64_t) ||
                                     our_item_val[1] == match->second))
      _intersection.emplace_back(our_item_idx);
  }
}

std::vector<std::string> PsiReceiver::output() {
  std::vector<std::string> output_;
  std::set<int> set_;
  for (auto idx : _intersection) {
    output_.emplace_back(_input[idx]);
    set_.emplace(idx);
  }
  return output_;
}
} // namespace psi
