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

#include "mpc_compare_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcCompareOp : public framework::OperatorWithKernel {

public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of MpcCompareOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                      platform::errors::NotFound(
                          "Input(Y) of MpcCompareOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of MpcCompareOp should not be null."));

    auto dim_x = ctx->GetInputDim("X");
    auto dim_y = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_GE(dim_x.size(), dim_y.size(),
                      "The size of dim_y should not be greater than dim_x's.");

    ctx->ShareDim("Y", /*->*/ "Out");
    ctx->ShareLoD("Y", /*->*/ "Out");
  }

  framework::OpKernelType
  GetExpectedKernelType(const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class MpcCompareOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of MpcCompareOp.");
    AddInput("Y", "(Tensor), The second input tensor of MpcCompareOp.");
    AddOutput("Out", "(Tensor), The output tensor of MpcCompareOp.");
    AddComment(R"DOC(
MPC Compare Operator.
)DOC");
  }
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(mpc_greater_than, ops::MpcCompareOp,
                             ops::MpcCompareOpMaker);
REGISTER_OP_CPU_KERNEL(
    mpc_greater_than,
    ops::MpcCompareOpKernel<paddle::platform::CPUDeviceContext, int64_t,
                            ops::MpcGreaterThanFunctor>);

REGISTER_OP_WITHOUT_GRADIENT(mpc_greater_equal, ops::MpcCompareOp,
                             ops::MpcCompareOpMaker);
REGISTER_OP_CPU_KERNEL(
    mpc_greater_equal,
    ops::MpcCompareOpKernel<paddle::platform::CPUDeviceContext, int64_t,
                            ops::MpcGreaterEqualFunctor>);

REGISTER_OP_WITHOUT_GRADIENT(mpc_less_than, ops::MpcCompareOp,
                             ops::MpcCompareOpMaker);
REGISTER_OP_CPU_KERNEL(
    mpc_less_than, ops::MpcCompareOpKernel<paddle::platform::CPUDeviceContext,
                                           int64_t, ops::MpcLessThanFunctor>);

REGISTER_OP_WITHOUT_GRADIENT(mpc_less_equal, ops::MpcCompareOp,
                             ops::MpcCompareOpMaker);
REGISTER_OP_CPU_KERNEL(
    mpc_less_equal, ops::MpcCompareOpKernel<paddle::platform::CPUDeviceContext,
                                            int64_t, ops::MpcLessEqualFunctor>);

REGISTER_OP_WITHOUT_GRADIENT(mpc_equal, ops::MpcCompareOp,
                             ops::MpcCompareOpMaker);
REGISTER_OP_CPU_KERNEL(
    mpc_equal, ops::MpcCompareOpKernel<paddle::platform::CPUDeviceContext,
                                       int64_t, ops::MpcEqualFunctor>);

REGISTER_OP_WITHOUT_GRADIENT(mpc_not_equal, ops::MpcCompareOp,
                             ops::MpcCompareOpMaker);
REGISTER_OP_CPU_KERNEL(
    mpc_not_equal, ops::MpcCompareOpKernel<paddle::platform::CPUDeviceContext,
                                           int64_t, ops::MpcNotEqualFunctor>);
