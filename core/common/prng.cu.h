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

#pragma once

#include "aes.cu.h"

namespace common {

class PseudorandomNumberGenerator {

public:

    PseudorandomNumberGenerator();

    PseudorandomNumberGenerator(const void* seed, u32 seed_len);

    ~PseudorandomNumberGenerator();

    PseudorandomNumberGenerator(
        const PseudorandomNumberGenerator &other) = delete;

    PseudorandomNumberGenerator &operator=(
        const PseudorandomNumberGenerator &other) = delete;

    void set_seed(const void* seed, u32 seed_len);

    void get_array(void* res, size_t len, cudaStream_t stream = NULL);

    void xor_array(void* res, size_t len, cudaStream_t stream = NULL);

    void array_sub64(void* res, size_t len, cudaStream_t stream = NULL);

private:

    template <typename Func, typename Func2>
    void get_array_impl(Func aes_mode_func,
                        Func2 set_output_func, void* res, size_t len, cudaStream_t stream);

private:

    AES _aes;

    // byte size
    constexpr static const size_t _s_buffer_size = 2 * AES::blockSize();

    u32* _buffer;

    u32 _ctr;
};

using PRNG = PseudorandomNumberGenerator;

struct CharArrayBlock {
    // 128 bit
    char arr[16];

    CharArrayBlock() : arr{0} {}
};

using block = CharArrayBlock;

extern const block g_zero_block;

} // namespace common

