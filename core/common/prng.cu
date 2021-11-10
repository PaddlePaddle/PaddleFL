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

namespace common {

const block g_zero_block = CharArrayBlock();

PseudorandomNumberGenerator::PseudorandomNumberGenerator()
    : PseudorandomNumberGenerator(&g_zero_block, sizeof(g_zero_block)) {}

PseudorandomNumberGenerator::PseudorandomNumberGenerator(const void* seed, u32 seed_len)
    : _aes(seed, seed_len), _ctr(0) {

    cudaMalloc((void**)&_buffer, _s_buffer_size);

}

PseudorandomNumberGenerator::~PseudorandomNumberGenerator() {
    cudaFree(_buffer);
}

void PseudorandomNumberGenerator::set_seed(const void* seed, u32 seed_len) {
    _aes.makeKey(seed, seed_len);
    _ctr = 0;
}

__global__ void set_counter(u32 *pt, const u32 ctr, size_t block_num) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < block_num;
         i += blockDim.x * gridDim.x) {
        const size_t offset = i * (AES_BLOCKSIZE / 4);
        pt[offset + 0] = 0;
        pt[offset + 1] = 0;
        pt[offset + 2] = 0;
        pt[offset + 3] = ctr + i;
    }
}

template <typename Func, typename Func2>
void PseudorandomNumberGenerator::get_array_impl(
    Func aes_mode_func, Func2 set_output_func, void* res, size_t len, cudaStream_t stream) {

    size_t aes_block_num = len / _aes.blockSize();
    size_t remainder = len % _aes.blockSize();

    set_counter<<<1, 1, 0, stream>>>(_buffer, _ctr, 1);
    // cudaStreamSynchronize(stream);
    ((&_aes)->*aes_mode_func)(_buffer, res, aes_block_num, stream);

    _ctr += aes_block_num;
   reinterpret_cast<byte*&>(res) += aes_block_num * _aes.blockSize();

    if (remainder) {
        set_counter<<<1, 1, 0, stream>>>(_buffer, _ctr++, 1);
        // cudaStreamSynchronize(stream);
        // 4 32-bit words for an aes block
        // reuse _buffer + 4 as ciphertext buffer
        _aes.encrypt_ctr(_buffer, _buffer + 4, 1, stream);
        set_output_func<<<1, 1, 0, stream>>>(res, _buffer + 4, remainder);
        // cudaStreamSynchronize(stream);
    }
}

__global__ void bytes_copy(void* dest, const void* src, size_t size) {
    for (u32 i = 0; i < size; ++i) {
        reinterpret_cast<byte*>(dest)[i] = reinterpret_cast<const byte*>(src)[i];
    }
}

__global__ void bytes_xor(void* dest, const void* src, size_t size) {
    for (u32 i = 0; i < size; ++i) {
        reinterpret_cast<byte*>(dest)[i] ^= reinterpret_cast<const byte*>(src)[i];
    }
}

// only for process 64 bits from 128 bits
__global__ void __sub64(void* dest, const void* src, size_t size) {
    reinterpret_cast<uint64_t*>(dest)[0] =
        reinterpret_cast<uint64_t*>(dest)[0] - reinterpret_cast<const uint64_t*>(src)[0];
}

void PseudorandomNumberGenerator::get_array(void* res, size_t len, cudaStream_t stream) {
    get_array_impl(&AES::encrypt_ctr, bytes_copy, res, len, stream);
}

void PseudorandomNumberGenerator::xor_array(void* res, size_t len, cudaStream_t stream) {
    get_array_impl(&AES::encrypt_ctr_xor, bytes_xor, res, len, stream);
}

void PseudorandomNumberGenerator::array_sub64(void* res, size_t len, cudaStream_t stream) {
    get_array_impl(&AES::encrypt_ctr_sub64, __sub64, res, len, stream);
}

} // namespace common

