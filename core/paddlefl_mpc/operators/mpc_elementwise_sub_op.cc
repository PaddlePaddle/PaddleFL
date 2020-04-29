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

#include "mpc_elementwise_sub_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class MpcElementwiseSubOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound(
            "Input(X) of MpcElementwiseSubOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::NotFound(
            "Input(Y) of MpcElementwiseSubOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound(
            "Output(Out) of MpcElementwiseSubOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->GetInputDim("X"), ctx->GetInputDim("Y"),
        platform::errors::InvalidArgument(
            "The dimensions of X should be equal with the dimensions of Y. "
            "But received the dimensions of X is [%s], the dimensions of Y is "
            "[%s]",
            ctx->GetInputDim("X"), ctx->GetInputDim("Y")));

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MpcElementwiseSubOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "(Tensor), The first input tensor of mpc elementwise sub op.");
    AddInput("Y",
             "(Tensor), The second input tensor of mpc elementwise sub op.");
    AddOutput("Out", "(Tensor), The output tensor of mpc elementwise sub op.");
    AddComment(R"DOC(
MPC elementwise sub Operator.
)DOC");
  }
};

class MpcElementwiseSubGradOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true, "Input(Y) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput(out_grad_name), true,
                      "Input(Out@GRAD) should not be null.");
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", /*->*/ x_grad_name);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", /*->*/ y_grad_name);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);
    }
  }
};

template <typename T>
class MpcElementwiseSubGradMaker : public framework::SingleGradOpDescMaker {
public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> retv(new T());
    retv->SetType("mpc_elementwise_sub_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
    return retv;
  }
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mpc_elementwise_sub, ops::MpcElementwiseSubOp,
                  ops::MpcElementwiseSubOpMaker,
                  ops::MpcElementwiseSubGradMaker<paddle::framework::OpDesc>);

REGISTER_OPERATOR(mpc_elementwise_sub_grad, ops::MpcElementwiseSubGradOp);

REGISTER_OP_CPU_KERNEL(
    mpc_elementwise_sub,
    ops::MpcElementwiseSubKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(mpc_elementwise_sub_grad,
                       ops::MpcElementwiseSubGradKernel<
                           paddle::platform::CPUDeviceContext, int64_t>);
