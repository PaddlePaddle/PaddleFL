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

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"

#include "../abstract_network.h"

namespace paddle {
namespace mpc {

__global__ void cuda_cpy(void*dest, const void* src, size_t n) {
     for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
          i < n; i += blockDim.x * gridDim.x) {
         reinterpret_cast<char*>(dest)[i] =
             reinterpret_cast<const char*>(src)[i];
     }
}

// A full-connected network based on cuda memcpy.
// only for cuda local ut.
class CudaCopyNetwork : public AbstractNetwork {
public:

    CudaCopyNetwork(const size_t party_id, const size_t net_size,
                    cudaStream_t stream = NULL)
        : _n_ranks(net_size), _my_rank(party_id), _stream(stream) {}

    ~CudaCopyNetwork() {}

    static int get_buf_id(int from, int to) {
        //  [0]: 0 -> 1
        //  [1]: 0 -> 2
        //  [2]: 1 -> 2
        //  [3]: 1 -> 0
        //  [4]: 2 -> 0
        //  [5]: 2 -> 1
        return from * 2 + (((to + 3 - from) % 3) == 1 ? 0 : 1);
    }

    void send(size_t party, const void *data, size_t size) override {

        int buf_id = get_buf_id(party_id(), party);
        int lock_id = buf_id;
        std::unique_lock<std::mutex> lk(_s_m[lock_id]);
        _s_cv[lock_id].wait(lk, [buf_id]{ return !_s_has_data[buf_id]; });

        _s_data_buff[buf_id] = data;
        _s_buff_len[buf_id] = size;
        _s_has_data[buf_id] = 1;
        lk.unlock();
        _s_cv[lock_id].notify_one();

    }

    void recv(size_t party, void *data, size_t size) override {

        int buf_id = get_buf_id(party, party_id());
        int lock_id = buf_id;
        std::unique_lock<std::mutex> lk(_s_m[lock_id]);
        _s_cv[lock_id].wait(lk, [buf_id]{ return _s_has_data[buf_id]; });

        if (size != _s_buff_len[buf_id]) {
            throw std::runtime_error("unexpected msg len: " + std::to_string(size)
                                     + "vs " + std::to_string(_s_buff_len[buf_id]) + "in buf");
        }

        {
            std::unique_lock<std::mutex> lk(_s_m_);
            copy(data, _s_data_buff[buf_id], size);
        }

        _s_has_data[buf_id] = 0;
        lk.unlock();
        _s_cv[lock_id].notify_one();
    }

    size_t party_id() const override {
        return _my_rank;
    }

    size_t party_num() const override {
        return _n_ranks;
    }

    void init() override {}

private:
    void copy(void* dest, const void* src, size_t n) {
#define __CUDA_COPY_THREAD_SIZE 512
        dim3 block_size = dim3(__CUDA_COPY_THREAD_SIZE, 1);
        dim3 grid_size = dim3((n + __CUDA_COPY_THREAD_SIZE - 1) / __CUDA_COPY_THREAD_SIZE, 1);
        cuda_cpy<<<grid_size, block_size, 0, _stream>>>(dest, src, n);
        cudaStreamSynchronize(_stream);
    }

private:
    int _n_ranks;
    int _my_rank;
    int _net_size;
    cudaStream_t _stream;

    static std::mutex _s_m[6];
    static std::mutex _s_m_;
    static std::condition_variable _s_cv[6];

    static bool _s_has_data[6];
    static const void* _s_data_buff[6];
    static size_t _s_buff_len[6];
};

// only used in seperated ut, defined here may not cause linking error
std::mutex CudaCopyNetwork::_s_m[6];
std::mutex CudaCopyNetwork::_s_m_;
std::condition_variable CudaCopyNetwork::_s_cv[6];

bool CudaCopyNetwork::_s_has_data[6] = { 0 };
const void* CudaCopyNetwork::_s_data_buff[6] = { nullptr };
size_t CudaCopyNetwork::_s_buff_len[6] = { 0 };

} // namespace mpc
} // namespace paddle
