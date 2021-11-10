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

#include "aes.cu.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

#include "gtest/gtest.h"

namespace common {

class AESTest : public ::testing::Test {
public:
    const char* _plain = "\x00\x11\x22\x33\x44\x55\x66\x77"
        "\x88\x99\xaa\xbb\xcc\xdd\xee\xff";

    const char* _key = "\x00\x01\x02\x03\x04\x05\x06\x07"
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
        "\x10\x11\x12\x13\x14\x15\x16\x17"
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f";

    const char* _cipher0 = "\x69\xc4\xe0\xd8\x6a\x7b\x04\x30"
        "\xd8\xcd\xb7\x80\x70\xb4\xc5\x5a";

    const char* _cipher1 = "\xdd\xa9\x7c\xa4\x86\x4c\xdf\xe0"
        "\x6e\xaf\x70\xa0\xec\x0d\x71\x91";

    const char* _cipher2 = "\x8e\xa2\xb7\xca\x51\x67\x45\xbf"
        "\xea\xfc\x49\x90\x4b\x49\x60\x89";

    const size_t _block_size = 16;
    const size_t _bench_size = 0x1p22;

    std::string _ct[3];

    uint8_t* _d_pt;
    uint8_t* _d_ct;

    uint8_t _h_ct[16];

    void SetUp() {
        _ct[0] = std::string(_cipher0, _block_size);
        _ct[1] = std::string(_cipher1, _block_size);
        _ct[2] = std::string(_cipher2, _block_size);

        cudaMalloc((void**)&_d_pt, _block_size);
        cudaMalloc((void**)&_d_ct, _block_size);

        cudaMemcpy(_d_pt, _plain, _block_size, cudaMemcpyHostToDevice);
    }

    void TearDown() {
        cudaFree(_d_pt);
        cudaFree(_d_ct);
    }

    void bench_ctr(int key_size) {
        uint8_t* ct;

        cudaMalloc((void**)&ct, _bench_size * _block_size);

        AES aes(_key, key_size);

        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();

        aes.encrypt_ctr(_d_pt, ct, _bench_size);
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
        std::cerr << d.count() * 1.0 / _bench_size << " ns per aes block\n";
        std::cerr << 128 / (d.count() * 1.0 / _bench_size) << " Gbps\n";

        t0 = std::chrono::high_resolution_clock::now();

        aes.encrypt_ctr_sub64(_d_pt, ct, _bench_size);
        cudaDeviceSynchronize();

        t1 = std::chrono::high_resolution_clock::now();
        d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
        std::cerr << d.count() * 1.0 / _bench_size << " ns per aes block with sub\n";
        std::cerr << 128 / (d.count() * 1.0 / _bench_size) << " Gbps\n";

        t0 = std::chrono::high_resolution_clock::now();

        aes.encrypt_ctr_xor(_d_pt, ct, _bench_size);
        cudaDeviceSynchronize();

        t1 = std::chrono::high_resolution_clock::now();
        d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
        std::cerr << d.count() * 1.0 / _bench_size << " ns per aes block with xor\n";
        std::cerr << 128 / (d.count() * 1.0 / _bench_size) << " Gbps\n";

        cudaFree(ct);
    }

    void bench_ecb(int key_size) {
        uint8_t* pt;
        uint8_t* ct;

        cudaMalloc((void**)&pt, _bench_size * _block_size);
        cudaMalloc((void**)&ct, _bench_size * _block_size);

        AES aes(_key, key_size);

        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();

        aes.encrypt_ecb(pt, ct, _bench_size);
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
        std::cerr << d.count() * 1.0 / _bench_size << " ns per aes block\n";
        std::cerr << 128 / (d.count() * 1.0 / _bench_size) << " Gbps\n";

        cudaFree(pt);
        cudaFree(ct);
    }
};

TEST_F(AESTest, base_test128) {
    AES aes(_key, 128);

    aes.encrypt(_d_pt, _d_ct);

    cudaMemcpy(_h_ct, _d_ct, _block_size, cudaMemcpyDeviceToHost);

    EXPECT_EQ(_ct[0], std::string(reinterpret_cast<const char*>(_h_ct), _block_size));

    EXPECT_EQ(128, aes.blockBits());
    EXPECT_EQ(16, aes.blockSize());
    EXPECT_EQ(128, aes.keyBits());
    EXPECT_EQ(16, aes.keySize());
}

TEST_F(AESTest, base_test192) {
    AES aes(_key, 192);

    aes.encrypt(_d_pt, _d_ct);

    cudaMemcpy(_h_ct, _d_ct, _block_size, cudaMemcpyDeviceToHost);

    EXPECT_EQ(_ct[1], std::string(reinterpret_cast<const char*>(_h_ct), _block_size));

    EXPECT_EQ(128, aes.blockBits());
    EXPECT_EQ(16, aes.blockSize());
    EXPECT_EQ(192, aes.keyBits());
    EXPECT_EQ(24, aes.keySize());
}

TEST_F(AESTest, base_test256) {
    AES aes(_key, 256);

    aes.encrypt(_d_pt, _d_ct);

    cudaMemcpy(_h_ct, _d_ct, _block_size, cudaMemcpyDeviceToHost);

    EXPECT_EQ(_ct[2], std::string(reinterpret_cast<const char*>(_h_ct), _block_size));

    EXPECT_EQ(128, aes.blockBits());
    EXPECT_EQ(16, aes.blockSize());
    EXPECT_EQ(256, aes.keyBits());
    EXPECT_EQ(32, aes.keySize());
}

TEST_F(AESTest, key_size) {
    AES aes(_key, 32);

    aes.makeKey(_key, 16);
    EXPECT_EQ(16, aes.keySize());
    aes.encrypt(_d_pt, _d_ct);
    cudaMemcpy(_h_ct, _d_ct, _block_size, cudaMemcpyDeviceToHost);
    EXPECT_EQ(_ct[0], std::string(reinterpret_cast<const char*>(_h_ct), _block_size));

    aes.makeKey(_key, 24);
    EXPECT_EQ(24, aes.keySize());
    aes.encrypt(_d_pt, _d_ct);
    cudaMemcpy(_h_ct, _d_ct, _block_size, cudaMemcpyDeviceToHost);
    EXPECT_EQ(_ct[1], std::string(reinterpret_cast<const char*>(_h_ct), _block_size));

    aes.makeKey(_key, 32);
    EXPECT_EQ(32, aes.keySize());
    aes.encrypt(_d_pt, _d_ct);
    cudaMemcpy(_h_ct, _d_ct, _block_size, cudaMemcpyDeviceToHost);
    EXPECT_EQ(_ct[2], std::string(reinterpret_cast<const char*>(_h_ct), _block_size));
}

TEST_F(AESTest, invalid_key_size) {
    EXPECT_THROW({AES aes(_key, 17);} , std::invalid_argument);
}

TEST_F(AESTest, bench_ctr128) {
    bench_ctr(128);
}
TEST_F(AESTest, bench_ctr192) {
    bench_ctr(192);
}
TEST_F(AESTest, bench_ctr256) {
    bench_ctr(256);
}

TEST_F(AESTest, bench_ecb128) {
    bench_ecb(128);
}

TEST_F(AESTest, bench_ecb192) {
    bench_ecb(192);
}

TEST_F(AESTest, bench_ecb256) {
    bench_ecb(256);
}

TEST_F(AESTest, base_test128_sub) {
    AES aes(_key, 128);

    uint64_t foo[2] = {1234, 5678};

    cudaMemcpy(_d_ct, foo, _block_size, cudaMemcpyHostToDevice);

    aes.encrypt_ctr_sub64(_d_pt, _d_ct, 1);

    uint64_t bar[2];

    cudaMemcpy(bar, _d_ct, _block_size, cudaMemcpyDeviceToHost);

    uint64_t exp_bar[2];

    std::memcpy(exp_bar, _ct[0].data(), 16);

    exp_bar[0] = 1234 - exp_bar[0];
    exp_bar[1] = 5678 - exp_bar[1];

    // EXPECT_EQ(_ct[0], std::string(reinterpret_cast<const char*>(_h_ct), _block_size));

    EXPECT_EQ(exp_bar[0], bar[0]);
    EXPECT_EQ(exp_bar[1], bar[1]);
}

TEST_F(AESTest, base_test128_xor) {
    AES aes(_key, 128);

    uint64_t foo[2] = { 1234, 5678 };

    cudaMemcpy(_d_ct, foo, _block_size, cudaMemcpyHostToDevice);

    aes.encrypt_ctr_xor(_d_pt, _d_ct, 1);

    uint64_t bar[2];

    cudaMemcpy(bar, _d_ct, _block_size, cudaMemcpyDeviceToHost);

    uint64_t exp_bar[2];

    std::memcpy(exp_bar, _ct[0].data(), 16);

    exp_bar[0] = foo[0] ^ exp_bar[0];
    exp_bar[1] = foo[1] ^ exp_bar[1];

    EXPECT_EQ(exp_bar[0], bar[0]);
    EXPECT_EQ(exp_bar[1], bar[1]);
}
} // namespace common
