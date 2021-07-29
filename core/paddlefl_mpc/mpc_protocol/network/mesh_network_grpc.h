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

#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "grpc++/grpc++.h"
#include "paddle/fluid/platform/enforce.h"

#include "core/paddlefl_mpc/mpc_protocol/abstract_network.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_config.h"
#include "transport.grpc.pb.h"

namespace paddle {
namespace mpc {

class GrpcBuffer {
    public:

        GrpcBuffer() : buffer(std::vector<std::queue<std::string>>(MAX_NET_SIZE)) {}

        void write_buffer(const size_t party_id, const std::string data);

        void read_buffer(const size_t party_id, std::string& data);

    private:
        static const int BUFFER_LENGTH = 65535; // this version only supports BUFFER_LENGTH = 1.
        static const int MAX_NET_SIZE = 3;
        std::vector<std::queue<std::string>> buffer;

        std::mutex mtx;
        std::condition_variable write_cv;
        std::condition_variable read_cv;
};

//gRPC client impl
class TransportClient {
    public:
        TransportClient(){}
        TransportClient(std::shared_ptr<grpc::Channel> channel)
            : stub_(transport::Transport::NewStub(channel)) {}

        void send(const int party_id, const void* data, size_t size);

    private:
        std::unique_ptr<transport::Transport::Stub> stub_;
        static constexpr const size_t _max_retry = 3;
};

class MeshNetworkGrpc : public paddle::mpc::AbstractNetwork, public transport::Transport::Service {
    public:

        MeshNetworkGrpc(const size_t party_id, const size_t net_size, const std::string &endpoints)
            : _party_id(party_id), _net_size(net_size), _endpoints(endpoints),
            _is_initialized(false) {}

        ~MeshNetworkGrpc() {
            if (server != nullptr) {
                server->Shutdown();
            }
        }

        void send(size_t party, const void *data, size_t size) override;

        void recv(size_t party, void *data, size_t size) override;

        size_t party_id() const override { return _party_id; };

        size_t party_num() const override { return _net_size; };

        // must be called before use
        void init();

        grpc::Status send_data(grpc::ServerContext* context, const transport::GrpcRequest* request,
                       transport::GrpcReply* reply) override;
        // gRPC server: start server to listen
        void run_server(const std::string& server_address) {
            grpc::ServerBuilder builder;
            builder.SetMaxMessageSize(INT_MAX);
            builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
            builder.RegisterService(this);
            server = builder.BuildAndStart();
            VLOG(3) << "Server listening on " << server_address;

            server->Wait();
        }

    private:
        const size_t _party_id;
        const size_t _net_size;
        const std::string _endpoints;
        bool _is_initialized;
        std::unique_ptr<grpc::Server> server;
        std::unordered_map<int, TransportClient> client_map;
        GrpcBuffer grpc_buffer;

        void split(const std::string& s, std::vector<std::string>& tokens, const char& delim = ' ') {
            tokens.clear();
            size_t lastPos = s.find_first_not_of(delim, 0);
            size_t pos = s.find(delim, lastPos);
            std::string s_;
            while (lastPos != std::string::npos) {
                s_ = s.substr(lastPos, pos - lastPos);
                s_.erase(0, s_.find_first_not_of(" "));
                s_.erase(s_.find_last_not_of(" ") + 1);
                tokens.emplace_back(s_);
                lastPos = s.find_first_not_of(delim, pos);
                pos = s.find(delim, lastPos);
            }
        }

};

} // mpc
} // paddle
