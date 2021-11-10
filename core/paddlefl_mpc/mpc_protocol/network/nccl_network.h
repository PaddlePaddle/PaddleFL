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

#include <stdexcept>
#include <string>

#include "cuda_runtime.h"
#include "nccl.h"

#include "../abstract_network.h"

namespace paddle {
namespace mpc {

#define NCCLCHECK(cmd) do {                                \
    ncclResult_t r = cmd;                                  \
    if (r!= ncclSuccess) {                                 \
        std::string msg("Failed, NCCL error " + std::string(__FILE__) + ":" \
                        + std::to_string(__LINE__)  + " '" \
                        + std::string(ncclGetErrorString(r)) + "'"); \
        throw std::runtime_error(msg);                     \
    }                                                      \
} while(0)

// A full-connected network based on NCCL.
// usage:
// root/party 0 generate id by get_nccl_id() and broadcast to others
// then bootstrap by id
// nccl socket behavior may be affacted by env NCCL_SOCKET_IFNAME,
// NCCL_COMM_ID etc.
// see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
class NcclNetwork : public AbstractNetwork {
public:

    NcclNetwork(const size_t party_id, const size_t net_size, ncclUniqueId id,
                cudaStream_t stream = NULL)
        : _n_ranks(net_size), _my_rank(party_id), _stream(stream) {
            NCCLCHECK(ncclCommInitRank(&_comm, _n_ranks, id, _my_rank));
    }

    ~NcclNetwork() {
        ncclCommDestroy(_comm);
    }

    void send(size_t party, const void *data, size_t size) override {
        // use char type
        NCCLCHECK(ncclSend(data, size, ncclChar, party, _comm, _stream));
        // cudaStreamSynchronize(_stream);
    }

    void recv(size_t party, void *data, size_t size) override {
        NCCLCHECK(ncclRecv(data, size, ncclChar, party, _comm, _stream));
    }

    static ncclUniqueId get_nccl_id() {
        ncclUniqueId id;
        ncclGetUniqueId(&id);
        return id;
    }

    size_t party_id() const override {
        return _my_rank;
    }

    size_t party_num() const override {
        return _n_ranks;
    }

    void init() override {}

private:
    int _n_ranks;
    int _my_rank;
    int _net_size;
    ncclComm_t _comm;
    cudaStream_t _stream;
};

} // namespace mpc
} // namespace paddle
