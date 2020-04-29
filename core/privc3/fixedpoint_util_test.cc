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

#include "fixedpoint_util.h"
#include "gtest/gtest.h"

namespace aby3 {

TEST(FixedPointUtil, int64_test) {
  double input = 5.1;
  long long *ret[3];
  for (int i = 0; i < 3; i++) {
    ret[i] = (long long *)malloc(sizeof(long long));
  }

  FixedPointUtil<long long, 32>::share(input, ret);

  double result = FixedPointUtil<long long, 32>::reveal(ret);

  for (int i = 0; i < 3; i++) {
    free(ret[i]);
  }

  EXPECT_LT(std::abs(input - result), 0.0001);
}

TEST(FixedPointUtil, int32_test) {
  float input = -10;
  long *ret[3];
  for (int i = 0; i < 3; i++) {
    ret[i] = (long *)malloc(sizeof(long));
  }

  FixedPointUtil<long, 16>::share(input, ret);

  double result = FixedPointUtil<long, 16>::reveal(ret);

  for (int i = 0; i < 3; i++) {
    free(ret[i]);
  }

  EXPECT_LT(std::abs(input - result), 0.0001);
}
}
