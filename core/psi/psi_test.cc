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

#include "gtest/gtest.h"

namespace psi {

class PsiTest : public ::testing::Test {

public:
  static const unsigned int _s_test_size = 1e3;
  PsiSender _sender;
  PsiReceiver _recver;
  std::set<std::string> _test_data;

public:
  PsiTest()
      // for triggering cuckoo stash
      : _sender(_s_test_size, _s_test_size * 0.9, _mm_set1_epi64x(0)),
        _recver(_s_test_size, _s_test_size * 0.9, _mm_set1_epi64x(1)) {}

  ~PsiTest() {}

  virtual void SetUp() {
    for (size_t i = 0; i < _s_test_size; ++i) {
      _test_data.emplace(std::to_string(i));
    }
  }
  virtual void TearDown() {}
};

TEST_F(PsiTest, psi_test) {
  // for Block512 as choices
  for (size_t i = 0; i < 512; ++i) {
    auto send = _recver.np_ot().send_pre(i);
    auto send_back = _sender.np_ot().recv(i, send);
    _recver.np_ot().send_post(i, send_back);
  }
  _sender.init_offline(_test_data);
  _recver.init_offline(_test_data);
  _sender.sync();
  _recver.sync();
  auto masks = _recver.send_masks(0, _recver.cuckoo_bins_num());
  _sender.recv_masks(0, _recver.cuckoo_bins_num(), masks);

  const auto oprf_len = _sender.oprf_output_len();

  auto ptr_to_vec = [oprf_len](const uint8_t *data, size_t len) {
    std::vector<std::string> output_buf;
    for (auto *ptr = data; ptr != data + len; ptr += oprf_len) {
      output_buf.emplace_back(std::string((char *)ptr, oprf_len));
    }
    return output_buf;
  };

  // idx for 3 hash functions, see cuckoo hash
  for (size_t idx = 0; idx < 3; ++idx) {
    auto data = _sender.send_oprf_outputs(idx);
    auto output = ptr_to_vec(data.data(), data.size());
    _recver.recv_oprf_outputs(idx, output);
  }

  // now process cuckoo stash
  for (size_t idx = 0; idx < _recver.stash_bins_num(); ++idx) {
    auto hash_idx = idx + 3;
    size_t bin_idx = idx + _recver.cuckoo_bins_num();

    auto masks = _recver.send_masks(bin_idx, bin_idx + 1);
    _sender.recv_masks(bin_idx, bin_idx + 1, masks);

    auto data = _sender.send_oprf_outputs(idx);
    auto output = ptr_to_vec(data.data(), data.size());

    _recver.recv_oprf_outputs(hash_idx, output);
  }

  auto rhs_vec = _recver.output();

  std::set<std::string> rhs;
  for (auto &s : rhs_vec) {
    rhs.emplace(s);
  }

  EXPECT_TRUE(_test_data == rhs);
}

} // namespace psi
