// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "mpc_relu_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

// forward op defination
class MpcReluOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Y", in_dims);
  }
};

// forward input & output defination
class MpcReluOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddOutput("Y", "Output of relu_op");
    AddComment(R"DOC(
Mpc Relu Operator.
)DOC");
  }
};

// backward op defination
class MpcReluGradOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }
};

// backward type, input & output defination
template <typename T>
class MpcReluGradMaker : public framework::SingleGradOpDescMaker {
public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<T> Apply() const override {
    auto *op = new T();
    op->SetType("mpc_relu_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    return std::unique_ptr<T>(op);
  }
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;

using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(mpc_relu, ops::MpcReluOp, ops::MpcReluOpMaker,
                  ops::MpcReluGradMaker<paddle::framework::OpDesc>);
REGISTER_OPERATOR(mpc_relu_grad, ops::MpcReluGradOp);
REGISTER_OP_CPU_KERNEL(mpc_relu, ops::MpcReluKernel<CPU, int64_t>);
REGISTER_OP_CPU_KERNEL(mpc_relu_grad, ops::MpcReluGradKernel<CPU, int64_t>);
