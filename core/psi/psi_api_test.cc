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

#include "psi_api.h"

#include <thread>

#include "gtest/gtest.h"

namespace psi {

class PsiAPITest : public ::testing::Test {
public:
  std::set<std::string> _input;

  int _port;

  static const int _s_test_size = 1e3;

public:
  PsiAPITest() {
    for (int i = 0; i < _s_test_size; ++i) {
      _input.emplace(std::to_string(i));
    }
    _port = 45818;
  }

  ~PsiAPITest() {}
};

TEST_F(PsiAPITest, full_test) {
  auto test_send = [this]() {
    // find valid port
    for (int ret = SOCKET_ERROR; ret == SOCKET_ERROR; ++_port) {
      ret = psi_send(_port, _input, nullptr);
    }
  };
  auto t_send = std::thread(test_send);

  std::vector<std::string> output;

  std::this_thread::sleep_for(std::chrono::seconds(1));
  psi_recv("127.0.0.1", _port, _input, &output, nullptr);

  t_send.join();

  std::set<std::string> out_set;
  for (auto &x : output) {
    out_set.emplace(x);
  }
  ASSERT_EQ(out_set, _input);
}

} // namespace psi
