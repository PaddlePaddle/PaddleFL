/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Description:

#include "paddle/fluid/framework/op_registry.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_instance.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_config.h"

namespace paddle {
namespace operators {

using mpc::MpcConfig;

class MpcInitOp : public framework::OperatorBase {
public:

    MpcInitOp(const std::string& type,
              const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs) : OperatorBase(type, inputs, outputs, attrs) {}

    void RunImpl(const framework::Scope &scope,
                 const platform::Place &dev_place) const override {
        auto role = Attr<int>("role");
        auto local_addr = Attr<std::string>("local_addr");
        auto net_server_addr = Attr<std::string>("net_server_addr");
        auto net_server_port = Attr<int>("net_server_port");
        auto endpoints = Attr<std::string>("endpoints");
        auto network_mode = Attr<std::string>("network_mode");

        MpcConfig _mpc_config;
        _mpc_config.set_int(MpcConfig::ROLE, role);
        _mpc_config.set(MpcConfig::LOCAL_ADDR, local_addr);
        _mpc_config.set(MpcConfig::NET_SERVER_ADDR, net_server_addr);
        _mpc_config.set_int(MpcConfig::NET_SERVER_PORT, net_server_port);
        _mpc_config.set(MpcConfig::ENDPOINTS, endpoints);
        _mpc_config.set(MpcConfig::NETWORK_MODE, network_mode);
        mpc::MpcInstance::init_instance(_mpc_config);
    }
};

class MpcInitOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {

        AddComment(R"DOC(
Mpc Init Operator.
)DOC");
        AddAttr<std::string>("protocol_name",
                                      "(string , default aby3)"
                                      "protocol name")
        .SetDefault({"aby3"});
        AddAttr<int>("role", "trainer role.").SetDefault(0);
        AddAttr<std::string>("local_addr",
                                      "(string, default localhost)"
                                      "local addr")
        .SetDefault({"localhost"});
        AddAttr<std::string>("net_server_addr",
                                      "(string, default localhost)"
                                      "net server addr")
        .SetDefault({"localhost"});
        AddAttr<int>("net_server_port", "net server port, default to 6539.").SetDefault(6539);
        AddAttr<std::string>("endpoints",
                                      "(string, default endpoints)"
                                      "endpoints")
        .SetDefault({"endpoints"});
        AddAttr<std::string>("network_mode",
                                      "(string, default gloo)"
                                      "network_mode")
        .SetDefault({"gloo"});
    }
};

class MpcInitOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    // init protocol_name.
    // Other ops can infer output's shape according to protocol_name, 
    // e.g., mpc_mean_op, mpc_mul_op.
    auto protocol_name = ctx->Attrs().Get<std::string>("protocol_name");
    mpc::MpcInstance::init_protocol_name(protocol_name);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    mpc_init, ops::MpcInitOp,
    ops::MpcInitOpMaker, ops::MpcInitOpShapeInference);

