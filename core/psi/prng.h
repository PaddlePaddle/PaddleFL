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

#pragma once

#include <array>

#include "aes.h"

namespace psi {

class PseudorandomNumberGenerator {

public:

    PseudorandomNumberGenerator() = default;

    PseudorandomNumberGenerator(const block &seed);

    PseudorandomNumberGenerator(
        const PseudorandomNumberGenerator &other) = delete;

    PseudorandomNumberGenerator &operator=(
        const PseudorandomNumberGenerator &other) = delete;

    void set_seed(const block &b);

    template <typename T>
    T get() {
        T data;
        get_array(&data, sizeof(T));
        return data;
    }

    void get_array(void* res, size_t len);

    // for std::shuffle
    typedef uint64_t result_type;

    constexpr static uint64_t min() {
        return 0;
    }

    constexpr static uint64_t max() {
        return -1ull;
    }

    uint64_t operator()() {
        return get<uint64_t>();
    }

private:

    // buffer num for aes cipher
    static const size_t _s_buffer_size = 0x10000;

    static const size_t _s_byte_capacity = _s_buffer_size * sizeof(block);

    std::array<block, _s_buffer_size> _buffer;

    std::array<block, _s_buffer_size> _ctr64;

    uint64_t _ctr;

    AES _aes;

    size_t _now_byte;

    void refill_buffer();
};
} // namespace psi

