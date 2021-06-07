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
// test
namespace privc {

TEST(FixedPointUtils, int64_test) {
  double input = 5.1;
  long long *ret[2];
  for (int i = 0; i < 2; i++) {
    ret[i] = (long long *)malloc(sizeof(long long));
  }

  FixedPointUtils<long long, 32>::share(input, ret);

  double result = FixedPointUtils<long long, 32>::reveal(ret);

  for (int i = 0; i < 2; i++) {
    free(ret[i]);
  }

  EXPECT_LT(std::abs(input - result), 0.0001);
}

TEST(FixedPointUtils, int32_test) {
  float input = -10;
  long *ret[2];
  for (int i = 0; i < 2; i++) {
    ret[i] = (long *)malloc(sizeof(long));
  }

  FixedPointUtils<long, 16>::share(input, ret);

  double result = FixedPointUtils<long, 16>::reveal(ret);

  for (int i = 0; i < 2; i++) {
    free(ret[i]);
  }

  EXPECT_LT(std::abs(input - result), 0.0001);
}
}
