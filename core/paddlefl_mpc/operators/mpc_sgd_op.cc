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

#include "mpc_sgd_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include <string>
namespace paddle {
namespace operators {

class MpcSGDOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext *ctx) const override {
      PADDLE_ENFORCE(ctx->HasInput("Param"),
                     "Input(Param) of MPCSGDOp should not be null.");
      PADDLE_ENFORCE(ctx->HasInput("Grad"),
                     "Input(Grad) of MPCSGDOp should not be null.");
      PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                     "Input(LearningRate) of MPCSGDOp should not be null.");
      PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                     "Output(ParamOut) of MPCSGDOp should not be null.");

      auto lr_dims = ctx->GetInputDim("LearningRate");
      PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                        "Maybe the Input variable LearningRate has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function.");
      PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                        "Learning rate should have 1 element");
      auto param_dim = ctx->GetInputDim("Param");
      if (ctx->GetInputsVarType("Grad")[0] ==
          framework::proto::VarType::LOD_TENSOR) {
        PADDLE_ENFORCE_EQ(
            param_dim, ctx->GetInputDim("Grad"),
            platform::errors::InvalidArgument(
                "MPCSGD Operator's input Param and Grad dimensions do not match. "
                "The Param %s shape is [%s], but the Grad %s shape is [%s].",
                ctx->Inputs("Param")[0], param_dim, ctx->Inputs("Grad")[0],
                ctx->GetInputDim("Grad")));
      }
      ctx->SetOutputDim("ParamOut", param_dim);
  }

protected:
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext &ctx) const override {
      auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Param");
      return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class MpcSGDOpInferVarType : public framework::VarTypeInference {
public:
    void operator()(framework::InferVarTypeContext *ctx) const override {
      auto in_var_type = ctx->GetInputType("Param");
      PADDLE_ENFORCE(in_var_type == framework::proto::VarType::SELECTED_ROWS ||
                         in_var_type == framework::proto::VarType::LOD_TENSOR,
                     "The input Var's type should be LoDtensor or SelectedRows,"
                     " but the received var(%s)'s type is %s",
                     ctx->InputVarName("Param"), in_var_type);
      ctx->SetOutputType("ParamOut", in_var_type);
  }
};

class MpcSGDOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
      AddInput("Param", "(Tensor or SelectedRows) Input parameter");
      AddInput("LearningRate", "(Tensor) Learning rate of MPCSGD");
      AddInput("Grad", "(Tensor or SelectedRows) Input gradient");
      AddOutput("ParamOut",
                "(Tensor or SelectedRows, same with Param) "
                "Output parameter, should share the same memory with Param");
      AddComment(R"DOC(

MPCSGD operator

This operator implements one step of the stochastic gradient descent algorithm.

$$param\_out = param - learning\_rate * grad$$

)DOC");
  }
};

template <typename T>
struct GetLearningRate<platform::CPUDeviceContext, T> {
    double operator()(const framework::Tensor* t) {
        return *t->data<T>();
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    mpc_sgd, ops::MpcSGDOp, ops::MpcSGDOpMaker,
    ops::MpcSGDOpInferVarType);

#ifndef USE_CUDA

REGISTER_OP_CPU_KERNEL(
    mpc_sgd,
    ops::MpcSGDOpKernel<paddle::platform::CPUDeviceContext, int64_t, float>);

#endif // USE_CUDA
