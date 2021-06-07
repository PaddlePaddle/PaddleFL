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

#include "core/paddlefl_mpc/mpc_protocol/network/mesh_network_grpc.h"

#include <vector>

namespace paddle {
namespace mpc {

void GrpcBuffer::write_buffer(const size_t party_id, const std::string data) {
    std::unique_lock<std::mutex> lck(mtx);
    while (buffer[party_id].size() == BUFFER_LENGTH) {
        write_cv.wait(lck);
    }

    buffer[party_id].push(data);
    if (buffer[party_id].size() == BUFFER_LENGTH) {
        read_cv.notify_all();
    }
}

void GrpcBuffer::read_buffer(const size_t party_id, std::string& data) {
    std::unique_lock<std::mutex> lck(mtx);
    while(buffer[party_id].size() == 0) {
        read_cv.wait(lck);
    }

    data = buffer[party_id].front();
    buffer[party_id].pop();
    if (buffer[party_id].size() == BUFFER_LENGTH - 1) {
        write_cv.notify_one();
    }
}

void TransportClient::send(const int party_id, const void* data, size_t size) {
    transport::GrpcRequest request;
    request.set_party_id(party_id);
    request.set_data(data, size);

    transport::GrpcReply reply;
    grpc::ClientContext context;

    for (size_t i = 0; i < _max_retry; ++i) {
        grpc::Status status = stub_->send_data(&context, request, &reply);
        if (status.ok()) {
            return;
        }
        if (i == _max_retry) {
            VLOG(3) << "return code: " << status.error_code() << ", error_message: " << status.error_message();
            PADDLE_THROW(platform::errors::Fatal(
                "error: num of send retry times exceed. return code: [%d], error message: [%s]",
                status.error_code(), status.error_message()));
        }
        if (errno != EAGAIN) {
            PADDLE_THROW(platform::errors::Fatal("error: send, errno:: [%d].", errno));
        }
    }
}

// for test purpose
void MeshNetworkGrpc::init() {
    if (_is_initialized) {
        return;
    }

    // start grpc server
    std::vector<std::string> endpoints_vec;
    const char delim = ';';
    split(_endpoints, endpoints_vec, delim);
    std::thread grpc_(&MeshNetworkGrpc::run_server, this, endpoints_vec[party_id()]);
    grpc_.detach();

    // init grpc client
    for (size_t i = 0; i < endpoints_vec.size(); ++i) {
        if (party_id() != i) {
            auto channel = grpc::CreateChannel(endpoints_vec[i], grpc::InsecureChannelCredentials());

            PADDLE_ENFORCE(channel->WaitForConnected(gpr_time_add(
                           gpr_now(GPR_CLOCK_REALTIME),
                           gpr_time_from_seconds(100, GPR_TIMESPAN))),
                           "Failed to connect server [%s]", endpoints_vec[i]);

            client_map.insert(std::make_pair(i, TransportClient(channel)));
        }
    }

    _is_initialized = true;
}

grpc::Status MeshNetworkGrpc::send_data(grpc::ServerContext* context, const transport::GrpcRequest* request,
                       transport::GrpcReply* reply) {
    // receive data from client and write into buffer
    grpc_buffer.write_buffer(request->party_id(), request->data());
    reply->set_ret_code(0);
    return grpc::Status::OK;
}


void MeshNetworkGrpc::send(size_t party, const void *data, size_t size) {
    PADDLE_ENFORCE_NOT_NULL(data);
    PADDLE_ENFORCE(_is_initialized);
    PADDLE_ENFORCE_LT(party, _net_size, "Input role should be less than net_size.");
    PADDLE_ENFORCE_NE(party, _party_id, "Party should not send data to itself.");
    client_map[party].send(_party_id, data, size);
}

void MeshNetworkGrpc::recv(size_t party, void *data, size_t size) {
    PADDLE_ENFORCE_NOT_NULL(data);
    PADDLE_ENFORCE(_is_initialized);
    PADDLE_ENFORCE_NE(party, _party_id, "Party should not receive data from itself.");

    std::string recv_data;
    grpc_buffer.read_buffer(party, recv_data);
    memcpy(data, (char*)recv_data.c_str(), size);
}

} // mpc
} // paddle
