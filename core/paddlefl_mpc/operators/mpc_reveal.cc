/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "core/paddlefl_mpc/operators/mpc_reveal_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {

class MpcRevealOp : public framework::OperatorWithKernel {
 public:
  MpcRevealOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "reveal");

    auto x_dims = ctx->GetInputDim("X");
    std::vector<int64_t> output_dims;
    for (size_t i = 1; i < x_dims.size(); i++) {
      output_dims.push_back(x_dims[i]);
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MpcRevealOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of reveal operator.");
    AddOutput("Out", "(Tensor) Output tensor of reveal operator.");
    AddComment(R"DOC(
**reveal operator**
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(mpc_reveal, ops::MpcRevealOp, ops::MpcRevealOpMaker);
REGISTER_OP_CPU_KERNEL(
    mpc_reveal,
    ops::MpcRevealKernel<paddle::platform::CPUDeviceContext, int64_t>);
