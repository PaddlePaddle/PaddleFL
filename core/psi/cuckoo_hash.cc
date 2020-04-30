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

#include "cuckoo_hash.h"

namespace psi {
CuckooHasher::CuckooHasher(size_t input_size)
    : _bin_num(1.2 * input_size), _cuckoo_size(input_size) {
  _bins.resize(_bin_num);
}
void CuckooHasher::insert_item(
    size_t item_idx, const std::array<std::vector<block>, 4> &hash_tab,
    size_t hash_idx, size_t tried) {

  size_t hashval =
      *reinterpret_cast<const size_t *>(&hash_tab[hash_idx][item_idx]);

  size_t addr = hashval % _bin_num;

  if (_bins[addr].is_empty()) {
    _bins[addr].item_idx = item_idx;
    _bins[addr].hash_idx = hash_idx;

  } else if (tried < _cuckoo_size) {
    size_t evict_item_idx = _bins[addr].item_idx;
    size_t evict_hash_idx = _bins[addr].hash_idx;

    _bins[addr].item_idx = item_idx;
    _bins[addr].hash_idx = hash_idx;
    // use block[0,1,2] as hash val
    insert_item(evict_item_idx, hash_tab, (evict_hash_idx + 1) % 3, tried + 1);
  } else {
    _stash.emplace_back(item_idx, hash_idx);
  }
}
void CuckooHasher::insert_all(
    const std::array<std::vector<block>, 4> &hash_tab) {
  for (size_t idx = 0; idx < hash_tab[0].size(); ++idx) {
    insert_item(idx, hash_tab);
  }
}

SimpleHasher::SimpleHasher(size_t other_size) : _bin_num(1.2 * other_size) {
  _table.resize(_bin_num);
}
void SimpleHasher::insert_all(
    const std::array<std::vector<block>, 4> &hash_tab) {
  for (size_t item_idx = 0; item_idx < hash_tab[0].size(); ++item_idx) {
    for (size_t hash_idx = 0; hash_idx < 3; ++hash_idx) {

      size_t hashval =
          *reinterpret_cast<const size_t *>(&hash_tab[hash_idx][item_idx]);
      size_t addr = hashval % _bin_num;

      _table[addr].emplace_back(item_idx, hash_idx);
    }
  }
}

} // namespace psi
