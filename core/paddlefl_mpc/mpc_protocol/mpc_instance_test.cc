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
// limitations under the License.i

#include "mpc_instance.h"
#include "mpc_config.h"
#include "gtest/gtest.h"
#include <thread>

#include "aby3_protocol.h"
#include "mpc_protocol_factory.h"
#include "gtest/gtest.h"

namespace paddle {
namespace mpc {

using namespace std;

TEST(MpcInstanceTest, InitInstance) {
  using paddle::platform::EnforceNotMet;

  EXPECT_THROW(MpcInstance::mpc_instance(), EnforceNotMet);

  auto gloo_store = std::make_shared<gloo::rendezvous::HashStore>();
  std::shared_ptr<std::thread> threads[3];
  for (int idx = 0; idx < 3; ++idx) {
    threads[idx] = std::make_shared<std::thread>([gloo_store, idx]() {
      const std::string protocol_name("aby3");
      MpcConfig aby3_config;
      aby3_config.set_int(Aby3Config::ROLE, idx);
      auto mpc_instance = MpcInstance::init_instance_with_store(
          protocol_name, aby3_config, gloo_store);
      ASSERT_NE(MpcInstance::mpc_instance(), nullptr);
      EXPECT_EQ(MpcInstance::mpc_instance(), mpc_instance);
      EXPECT_EQ(mpc_instance, MpcInstance::init_instance_with_store(
                                  protocol_name, aby3_config, gloo_store));
      EXPECT_EQ(mpc_instance->mpc_protocol()->name(), "aby3");
    });
  }
  EXPECT_THROW(MpcInstance::mpc_instance(), EnforceNotMet);

  for (auto thread : threads) {
    thread->join();
  }
}

} // mpc
} // paddle
