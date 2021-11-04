// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "prng.cu.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

#include "gtest/gtest.h"

namespace common {
TEST(prng, base) {
    char key[16] = { '\x80' };

    size_t size = 16;

    PRNG p(key, size);

    uint8_t* ct;
    uint8_t h_ct[16] = { 0 };
    cudaMalloc((void**)&ct, 16);
    cudaMemcpy(ct, h_ct, 16, cudaMemcpyHostToDevice);
    p.get_array(ct, 15);
    cudaMemcpy(h_ct, ct, 16, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // from aes test vector:
    // KEY = 80000000000000000000000000000000
    // PLAINTEXT = 00000000000000000000000000000000
    // CIPHERTEXT = 0edd33d3c621e546455bd8ba1418bec8
    EXPECT_EQ(std::string("\x0e\xdd\x33\xd3\xc6\x21\xe5\x46"
                          "\x45\x5b\xd8\xba\x14\x18\xbe"),
              std::string(reinterpret_cast<const char*>(h_ct), 15));
    EXPECT_EQ(0, h_ct[15]);
}

TEST(prng, bench_ctr) {
    const size_t bench_size = 0x1p22;

    const char* key = "\x00\x01\x02\x03\x04\x05\x06\x07"
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f";

    size_t size = 16;

    PRNG p(key, size);

    uint8_t* ct;
    cudaMalloc((void**)&ct, bench_size * size);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    p.get_array(ct, bench_size * size);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cerr << 128.0 / (d.count() * 1.0 / bench_size) << " Gbps\n";

    cudaFree(ct);
}


TEST(prng, bench_ctr_remainder) {
    const size_t bench_size = 0x1p22 + 1;

    const char* key = "\x00\x01\x02\x03\x04\x05\x06\x07"
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f";

    size_t size = 16;

    PRNG p(key, size);

    uint8_t* ct;
    cudaMalloc((void**)&ct, bench_size * size - 1);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    p.get_array(ct, bench_size * size - 1);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cerr << 128.0 / (d.count() * 1.0 / bench_size) << " Gbps\n";

    cudaFree(ct);
}

TEST(prng, sub64) {
    PRNG p(&g_zero_block, 16);

    uint64_t buf[3];

    uint8_t* ct;
    cudaMalloc((void**)&ct, 8 * 3);
    p.get_array(ct, 24);
    cudaMemcpy(buf, ct, 24, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    EXPECT_NE(0, buf[0]);
    EXPECT_NE(0, buf[1]);
    EXPECT_NE(0, buf[2]);
    // rewind key & seed
    p.set_seed(&g_zero_block, 16);
    p.array_sub64(ct, 24);

    cudaMemcpy(buf, ct, 24, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_EQ(0, buf[0]);
    EXPECT_EQ(0, buf[1]);
    EXPECT_EQ(0, buf[2]);
}

TEST(prng, xor) {
    PRNG p(&g_zero_block, 16);

    uint64_t buf[3];

    uint8_t* ct;
    cudaMalloc((void**)&ct, 8 * 3);
    p.get_array(ct, 24);
    cudaMemcpy(buf, ct, 24, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    EXPECT_NE(0, buf[0]);
    EXPECT_NE(0, buf[1]);
    EXPECT_NE(0, buf[2]);
    // rewind key & seed
    p.set_seed(&g_zero_block, 16);
    p.xor_array(ct, 24);

    cudaMemcpy(buf, ct, 24, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_EQ(0, buf[0]);
    EXPECT_EQ(0, buf[1]);
    EXPECT_EQ(0, buf[2]);
}
} // namespace common
