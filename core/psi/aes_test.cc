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

#include "aes.h"

#include <chrono>
#include <cstring>
#include <iostream>

#include "gtest/gtest.h"

#include "rand_utils.h"

namespace psi {

TEST(aes, base_test) {
    std::string plain("\x00\x11\x22\x33\x44\x55\x66\x77"
                      "\x88\x99\xaa\xbb\xcc\xdd\xee\xff", 16);

    std::string key("\x00\x01\x02\x03\x04\x05\x06\x07"
                    "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16);

    std::string cipher("\x69\xc4\xe0\xd8\x6a\x7b\x04\x30"
                       "\xd8\xcd\xb7\x80\x70\xb4\xc5\x5a", 16);

    block p;

    block k;

    block c;

    std::memcpy(&p, plain.data(), plain.size());

    std::memcpy(&k, key.data(), key.size());

    std::memcpy(&c, cipher.data(), cipher.size());

    AES aes(k);

    block c_ = aes.ecb_enc_block(p);

    EXPECT_TRUE(equals(c, c_));

    aes.ecb_enc_blocks(&p, 1, &c_);

    EXPECT_TRUE(equals(c, c_));
}

const size_t bench_size = 0x10000;

block p[bench_size];

block c[bench_size];

TEST(aes, bench) {
    std::string key("\x00\x01\x02\x03\x04\x05\x06\x07"
                    "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16);

    block k;

    std::memcpy(&k, key.data(), key.size());

    AES aes(k);

    const size_t rep = 0x100;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < rep; ++i) {
        aes.ecb_enc_blocks(p, bench_size, c);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cerr << d.count() * 1.0 / (rep * bench_size) << " ns per op\n";
}

} // namespace psi
