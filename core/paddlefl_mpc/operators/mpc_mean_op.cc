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

#include "paddle/fluid/framework/op_registry.h"
#include "mpc_mean_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcMeanOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::NotFound("Input(X) of MpcMeanOp should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasOutput("Out"), true,
            platform::errors::NotFound("Output(Out) of MpcMeanOp should not be null."));
        ctx->SetOutputDim("Out", {2, 1});
    }
};

class MpcMeanOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("X", "(Tensor), The first input tensor of mpc mean op.");
        AddOutput("Out", "(Tensor), The output tensor of mpc mean op.");
        AddComment(R"DOC(
MPC mean Operator calculates the mean of all elements in X.
)DOC");
    }
};

class MpcMeanOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
protected:
    std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
            const override {
        static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
        return m;
    }
};

class MpcMeanGradOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;
    using Tensor = framework::Tensor;
    
    void InferShape(framework::InferShapeContext *ctx) const override {
        ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
        ctx->ShareLoD("X", framework::GradVarName("X"));
    }

};

template <typename T>
class MpcMeanOpGradMaker : public framework::SingleGradOpMaker<T> {
public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
    void Apply(GradOpPtr<T> grad) const override {
        grad->SetType("mpc_mean_grad");
        grad->SetInput("X", this->Input("X"));
        grad->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
        grad->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mpc_mean, ops::MpcMeanOp,
                 ops::MpcMeanOpMaker, 
                 ops::MpcMeanOpInferVarType,
                 ops::MpcMeanOpGradMaker<paddle::framework::OpDesc>);

REGISTER_OPERATOR(mpc_mean_grad, ops::MpcMeanGradOp);

REGISTER_OP_CPU_KERNEL(
    mpc_mean, 
    ops::MpcMeanKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(
    mpc_mean_grad, 
    ops::MpcMeanGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
