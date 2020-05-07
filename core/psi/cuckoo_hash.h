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

#include <vector>

#include "utils.h"

namespace psi {

struct Bin {

  size_t item_idx;

  size_t hash_idx;

  Bin() : item_idx(-1), hash_idx(-1) {}

  Bin(size_t _item_idx, size_t _hash_idx)
      : item_idx(_item_idx), hash_idx(_hash_idx) {}

  bool is_empty() const { return item_idx == -1ull; }
};
class CuckooHasher {

  const size_t _bin_num;

  const size_t _cuckoo_size;

public:
  CuckooHasher(size_t input_size);

  CuckooHasher(const CuckooHasher &other) = delete;

  CuckooHasher &operator=(const CuckooHasher &other) = delete;

  std::vector<Bin> _bins;

  std::vector<Bin> _stash;

  void insert_item(size_t item_idx,
                   const std::array<std::vector<block>, 4> &hash_tab,
                   size_t hash_idx = 0, size_t tried = 0);

  void insert_all(const std::array<std::vector<block>, 4> &hash_tab);
};
class SimpleHasher {

  const size_t _bin_num;

public:
  // simple hasher is ownned by Alice, but size
  // is adjusted by Bob's input size
  SimpleHasher(size_t other_size);

  SimpleHasher(const SimpleHasher &other) = delete;

  SimpleHasher &operator=(const SimpleHasher &other) = delete;

  std::vector<std::vector<Bin>> _table;

  void insert_all(const std::array<std::vector<block>, 4> &hash_tab);
};

} // namespace psi
