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

#include <thread>

#include "aby3_protocol.h"
#include "privc_protocol.h"
#include "mpc_config.h"
#include "mpc_protocol_factory.h"
#include "gtest/gtest.h"

namespace paddle {
namespace mpc {

// tests for mpc_protocol and its configuration facilities

TEST(MpcProtocolTest, FindProtocol) {
  const std::string aby3_name("aby3");
  const std::string privc_name("privc");

  // empty name
  auto illegal = MpcProtocolFactory::build("");
  EXPECT_EQ(illegal, nullptr);

  // find aby3 with lower case name
  auto aby3_lower = MpcProtocolFactory::build("aby3");
  ASSERT_NE(aby3_lower, nullptr);
  EXPECT_EQ(aby3_lower->name(), aby3_name);

  // find aby3 with mixed lower and upper case name
  auto aby3_upper = MpcProtocolFactory::build("ABy3");
  ASSERT_NE(aby3_upper, nullptr);
  EXPECT_EQ(aby3_upper->name(), aby3_name);

  // find privc with lower case name
  auto privc_lower = MpcProtocolFactory::build("privc");
  ASSERT_NE(privc_lower, nullptr);
  EXPECT_EQ(privc_lower->name(), privc_name);

  // find privc with mixed lower and upper case name
  auto privc_upper = MpcProtocolFactory::build("PrivC");
  ASSERT_NE(privc_upper, nullptr);
  EXPECT_EQ(privc_upper->name(), privc_name);

  // find unknown protocol
  auto unknown = MpcProtocolFactory::build("foo");
  EXPECT_EQ(unknown, nullptr);
}

TEST(MpcProtocolTest, ProtocolInit) {
  using paddle::platform::EnforceNotMet;

  //aby3
  auto mpc = MpcProtocolFactory::build("aby3");
  ASSERT_NE(mpc, nullptr);

  // not yet initialized
  EXPECT_THROW(mpc->mpc_context(), EnforceNotMet);
  EXPECT_THROW(mpc->mpc_operators(), EnforceNotMet);
  EXPECT_THROW(mpc->network(), EnforceNotMet);

  // try initialize
  auto aby3 = std::dynamic_pointer_cast<Aby3Protocol>(mpc);
  MpcConfig config;

  // null store
  EXPECT_THROW(aby3->init_with_store(config, nullptr), EnforceNotMet);

  auto gloo_store = std::make_shared<gloo::rendezvous::HashStore>();
  std::shared_ptr<std::thread> threads[3];
  for (int idx = 0; idx < 3; ++idx) {
    threads[idx] = std::make_shared<std::thread>([gloo_store, idx]() {
      auto proto = std::make_shared<Aby3Protocol>();
      ASSERT_NE(proto, nullptr);
      MpcConfig aby3_config;
      aby3_config.set_int(MpcConfig::ROLE, idx);
      proto->init_with_store(aby3_config, gloo_store);

      ASSERT_NE(proto->network(), nullptr);
      EXPECT_EQ(proto->network()->party_id(), idx);
      EXPECT_EQ(proto->network()->party_num(), 3);

      ASSERT_NE(proto->mpc_context(), nullptr);
      EXPECT_EQ(proto->mpc_context()->next_party(), (idx + 1) % 3);
      EXPECT_EQ(proto->mpc_context()->pre_party(), (idx + 2) % 3);

      EXPECT_NE(proto->mpc_operators(), nullptr);
    });
  }

  for (auto thread : threads) {
    thread->join();
  }

  // privc
  mpc = MpcProtocolFactory::build("privc");
  ASSERT_NE(mpc, nullptr);

  auto privc = std::dynamic_pointer_cast<PrivCProtocol>(mpc);

  // null store
  EXPECT_THROW(privc->init_with_store(config, nullptr), EnforceNotMet);

  gloo_store = std::make_shared<gloo::rendezvous::HashStore>();
  std::shared_ptr<std::thread> privc_threads[2];
  for (int idx = 0; idx < 2; ++idx) {
    privc_threads[idx] = std::make_shared<std::thread>([gloo_store, idx]() {
      auto proto = std::make_shared<PrivCProtocol>();
      ASSERT_NE(proto, nullptr);
      MpcConfig privc_config;
      privc_config.set_int(MpcConfig::ROLE, idx);
      proto->init_with_store(privc_config, gloo_store);

      ASSERT_NE(proto->network(), nullptr);
      EXPECT_EQ(proto->network()->party_id(), idx);
      EXPECT_EQ(proto->network()->party_num(), 2);

      ASSERT_NE(proto->mpc_context(), nullptr);
      EXPECT_EQ(proto->mpc_context()->next_party(), (idx + 1) % 2);
      EXPECT_EQ(proto->mpc_context()->pre_party(), (idx + 1) % 2);

      EXPECT_NE(proto->mpc_operators(), nullptr);
    });
  }

  for (auto thread : privc_threads) {
    thread->join();
  }
}

TEST(MpcConfigTest, ConfigSetAndGet) {
  MpcConfig config;
  const std::string EMPTY_STR;
  const int ZERO = 0;

  // non-exist key leads to default string
  EXPECT_EQ(config.get("foo"), EMPTY_STR);

  // non-exist key leads to default int
  EXPECT_EQ(config.get_int("bar"), ZERO);

  // non-exist key leads to specified default str
  const std::string DEF_STR("default");
  EXPECT_EQ(config.get("foo1", DEF_STR), DEF_STR);

  // non-exist key leads to specified default int
  const int ONE = 1;
  EXPECT_EQ(config.get_int("foo2", ONE), ONE);

  const std::string KEY_STR("key1");
  const std::string KEY_INT("key2");
  const std::string VALUE_STR("value1");
  const int VALUE_INT = 2;
  config.set(KEY_STR, VALUE_STR).set_int(KEY_INT, VALUE_INT);

  // expected results
  EXPECT_EQ(config.get(KEY_STR), VALUE_STR);
  EXPECT_EQ(config.get_int(KEY_INT), VALUE_INT);

  // get wrong int
  EXPECT_THROW(config.get_int(KEY_STR), std::invalid_argument);

  // override existing key
  const std::string VALUE_STR2("value2");
  config.set(KEY_STR, VALUE_STR2);
  EXPECT_EQ(config.get(KEY_STR), VALUE_STR2);
  EXPECT_NE(config.get(KEY_STR), VALUE_STR);
}

} // mpc
} // paddle
