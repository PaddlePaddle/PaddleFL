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

#include "sse_transpose.h"

#include <array>

namespace psi {

void sse_load_sub_square(std::array<block, 2>& out, std::array<block, 128>& in,
                         size_t x, size_t y) {
    std::array<std::array<uint8_t, 16>, 2>& out_byte_view =
        *reinterpret_cast<std::array<std::array<uint8_t, 16>, 2> *>(&out);
    std::array<std::array<uint8_t, 16>, 128>& in_byte_view =
        *reinterpret_cast<std::array<std::array<uint8_t, 16>, 128> *>(&in);

    for (size_t l = 0; l < 16; l++) {
        out_byte_view[0][l] = in_byte_view[16 * x + l][2 * y];
        out_byte_view[1][l] = in_byte_view[16 * x + l][2 * y + 1];
    }
}

void sse_transpose_sub_square(std::array<block, 128>& out,
                              std::array<block, 2>& in, size_t x, size_t y) {
    std::array<std::array<uint16_t, 8>, 128>& out_u16_view =
        *reinterpret_cast<std::array<std::array<uint16_t, 8>, 128> *>(&out);

    for (size_t j = 0; j < 8; j++) {
        out_u16_view[16 * x + 7 - j][y] = _mm_movemask_epi8(in[0]);
        out_u16_view[16 * x + 15 - j][y] = _mm_movemask_epi8(in[1]);

        in[0] = _mm_slli_epi64(in[0], 1);
        in[1] = _mm_slli_epi64(in[1], 1);
    }
}

void sse_transpose128(std::array<block, 128>& in_out) {
    std::array<block, 2> a, b;

    for (size_t j = 0; j < 8; j++) {
        sse_load_sub_square(a, in_out, j, j);
        sse_transpose_sub_square(in_out, a, j, j);

        for (size_t k = 0; k < j; k++) {
            sse_load_sub_square(a, in_out, k, j);
            sse_load_sub_square(b, in_out, j, k);
            sse_transpose_sub_square(in_out, a, j, k);
            sse_transpose_sub_square(in_out, b, k, j);
        }
    }
}
} // namespace psi
