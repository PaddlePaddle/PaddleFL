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
#include "ot_extension.h"

#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "gtest/gtest.h"

#include "utils.h"

namespace common {

class OTtest : public ::testing::Test {

public:
  std::string _test_choices;

  static std::unique_ptr<NaorPinkasOTsender> _s_np_ot_sender;

  static std::unique_ptr<NaorPinkasOTreceiver> _s_np_ot_receiver;

  OTExtSender<Block512> _ot_ext_sender;

  OTExtReceiver<Block512> _ot_ext_receiver;

  Block512 _choices_blk;

  static const size_t _s_test_size = sizeof(_choices_blk) * 8;

  static const size_t _s_loop = 1e3;

public:
  OTtest() : _test_choices(_s_test_size / 8, '1') {
    std::memcpy(&_choices_blk, _test_choices.data(), _test_choices.size());
  }

  ~OTtest() {}

  void SetUp() {

    if (!_s_np_ot_sender && !_s_np_ot_receiver) {

      _s_np_ot_sender = std::unique_ptr<NaorPinkasOTsender>(
          new NaorPinkasOTsender(_s_test_size));

      _s_np_ot_receiver = std::unique_ptr<NaorPinkasOTreceiver>(
          new NaorPinkasOTreceiver(_s_test_size, _test_choices));

      for (size_t i = 0; i < _s_test_size; ++i) {

        auto send = _s_np_ot_sender->send_pre(i);

        auto send_back = _s_np_ot_receiver->recv(i, send);

        _s_np_ot_sender->send_post(i, send_back);
      }
    }
  }

  void TearDown() {}
};

std::unique_ptr<NaorPinkasOTsender> OTtest::_s_np_ot_sender;

std::unique_ptr<NaorPinkasOTreceiver> OTtest::_s_np_ot_receiver;

template <typename T> inline bool blk_eq(const T &b0, const T &b1) {
  auto *p0 = reinterpret_cast<const char *>(&b0);
  auto *p1 = reinterpret_cast<const char *>(&b1);

  return std::equal(p0, p0 + sizeof(T), p1);
}

TEST_F(OTtest, np_ot_test) {
  // NaorPinkas ot performed in SetUp
  for (size_t i = 0; i < _s_test_size; ++i) {
    int choice = _test_choices[i / 8] >> (i % 8) & 1 ? 1 : 0;
    EXPECT_TRUE(
        blk_eq(_s_np_ot_sender->_msgs[i][choice], _s_np_ot_receiver->_msgs[i]));
  }
}

TEST_F(OTtest, ot_ext_test) {

  _ot_ext_sender.init(_choices_blk, _s_np_ot_receiver->_msgs);
  _ot_ext_receiver.init(_s_np_ot_sender->_msgs);

  std::vector<Block512> send_msg(_s_loop);
  std::vector<std::array<Block512, 2>> recv_msg(_s_loop);

  _ot_ext_sender.fill_ot_buffer(send_msg);
  _ot_ext_receiver.fill_ot_buffer(recv_msg);

  for (size_t i = 0; i < _s_loop; ++i) {
    auto q = send_msg[i];
    auto t = recv_msg[i];

    auto rhs = q ^ ((t[0] ^ t[1]) & _choices_blk);
    bool res = blk_eq(t[0], rhs);
    ASSERT_TRUE(res);
  }
}

TEST_F(OTtest, ot_ext_test2) {

  _ot_ext_sender.init(_choices_blk, _s_np_ot_receiver->_msgs, true);
  _ot_ext_receiver.init(_s_np_ot_sender->_msgs, true);

  for (size_t i = 0; i < _s_loop; ++i) {
    auto q = _ot_ext_sender.get_ot_instance();
    auto t = _ot_ext_receiver.get_ot_instance();

    auto rhs = q ^ ((t[0] ^ t[1]) & _choices_blk);
    bool res = blk_eq(t[0], rhs);
    ASSERT_TRUE(res);
  }
}

} // namespace common
