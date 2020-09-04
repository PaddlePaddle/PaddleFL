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

#include "crypto.h"

#include <cstring>
#include <string>

#include "gtest/gtest.h"

namespace psi {

TEST(crypto, hash_block) {

    block in = ZeroBlock;

    for (size_t i = 0; i < 1e6; ++i) {
        hash_block(in);
    }
}

TEST(crypto, hash_blocks) {

    block in = ZeroBlock;

    for (size_t i = 0; i < 1e6; ++i) {
        hash_blocks({in, in});
    }
}

};
