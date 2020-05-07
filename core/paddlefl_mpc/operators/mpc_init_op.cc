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

// Description:

#include "paddle/fluid/framework/op_registry.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_config.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_instance.h"

namespace paddle {
namespace operators {

using mpc::MpcConfig;
using mpc::Aby3Config;

class MpcInitOp : public framework::OperatorBase {
public:
  MpcInitOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto protocol_name = Attr<std::string>("protocol_name");
    auto role = Attr<int>("role");
    auto local_addr = Attr<std::string>("local_addr");
    auto net_server_addr = Attr<std::string>("net_server_addr");
    auto net_server_port = Attr<int>("net_server_port");

    MpcConfig _mpc_config;
    _mpc_config.set_int(Aby3Config::ROLE, role);
    _mpc_config.set(Aby3Config::LOCAL_ADDR, local_addr);
    _mpc_config.set(Aby3Config::NET_SERVER_ADDR, net_server_addr);
    _mpc_config.set_int(Aby3Config::NET_SERVER_PORT, net_server_port);
    mpc::MpcInstance::init_instance(protocol_name, _mpc_config);
  }
};

class MpcInitOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {

    AddComment(R"DOC(
Where2 Operator.
)DOC");
    AddAttr<std::string>("protocol_name", "(string , default aby3)"
                                          "protocol name")
        .SetDefault({"aby3"});
    AddAttr<int>("role", "trainer role.").SetDefault(0);
    AddAttr<std::string>("local_addr", "(string, default localhost)"
                                       "local addr")
        .SetDefault({"localhost"});
    AddAttr<std::string>("net_server_addr", "(string, default localhost)"
                                            "net server addr")
        .SetDefault({"localhost"});
    AddAttr<int>("net_server_port", "net server port, default to 6539.")
        .SetDefault(6539);
  }
};

class MpcInitOpShapeInference : public framework::InferShapeBase {
public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(mpc_init, ops::MpcInitOp, ops::MpcInitOpMaker,
                  ops::MpcInitOpShapeInference);
