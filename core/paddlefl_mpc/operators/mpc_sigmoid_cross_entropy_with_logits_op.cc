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

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "mpc_sigmoid_cross_entropy_with_logits_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;
const int kIgnoreIndex = -100;

class MpcSigmoidCrossEntropyWithLogitsOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
        PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
        PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");

        auto x_dims = ctx->GetInputDim("X");
        auto labels_dims = ctx->GetInputDim("Label");

        int rank = x_dims.size();
        PADDLE_ENFORCE_EQ(rank, labels_dims.size(),
                          "Input(X) and Input(Label) shall have the same rank.");
        bool check = true;
        if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                    framework::product(labels_dims) <= 0)) {
            check = false;
        }

        if (check) {
            PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank),
                              framework::slice_ddim(labels_dims, 0, rank),
                             "Input(X) and Input(Label) shall have the same shape "
                             "except the last dimension.");
        }

        ctx->ShareDim("X", "Out");
        ctx->ShareLoD("X", "Out");
    }
};

class MpcSigmoidCrossEntropyWithLogitsGradOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
        PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
        PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                       "Input(Out@GRAD) shoudl be not null.");
        PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                       "Output(X@GRAD) should be not null.");

        auto x_dims = ctx->GetInputDim("X");
        auto labels_dims = ctx->GetInputDim("Label");
        auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

        int rank = x_dims.size();
        bool check = true;
        if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                    framework::product(labels_dims) <= 0)) {
            check = false;
        }

        if (check) {
            PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank),
                              framework::slice_ddim(labels_dims, 0, rank),
                              "Input(X) and Input(Label) shall have the same shape.");

            PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank),
                              framework::slice_ddim(dout_dims, 0, rank),
                              "Input(X) and Input(Out@Grad) shall have the same shape.");
        }

        ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
};

class MpcSigmoidCrossEntropyWithLogitsOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("X",
                 "(Tensor, default Tensor<float>), a 2-D tensor with shape N x D, "
                 "where N is the batch size and D is the number of classes. "
                 "This input is a tensor of logits computed by the previous "
                 " operator. Logits are unscaled log probabilities given as "
                 "log(p/(1-p)).");
        AddInput("Label",
                 "(Tensor, default Tensor<float>), a 2-D tensor of the same type "
                 "and shape as X. This input is a tensor of probabalistic labels "
                 "for each logit");
        AddOutput("Out",
                  "(Tensor, default Tensor<float>), a 2-D tensor with shape N x D "
                  " of elementwise logistic losses.");
        AddComment(R"DOC(
MpcSigmoidCrossEntropyWithLogits Operator.
)DOC");
    }
};

template <typename T>
class MpcSigmoidCrossEntropyWithLogitsGradOpMaker : public framework::SingleGradOpMaker<T> {
public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
    void Apply(GradOpPtr<T> grad) const override {
        grad->SetType("mpc_sigmoid_cross_entropy_with_logits_grad");
        grad->SetInput("X", this->Input("X"));
        grad->SetInput("Label", this->Input("Label"));
        grad->SetInput("Out", this->Output("Out"));
        grad->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
        grad->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
        grad->SetAttrMap(this->Attrs());
    }
};

DECLARE_INPLACE_OP_INFERER(MpcSigmoidCrossEntropyWithLogitsInplaceInferer,
                           {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(MpcSigmoidCrossEntropyWithLogitsGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    mpc_sigmoid_cross_entropy_with_logits, ops::MpcSigmoidCrossEntropyWithLogitsOp,
    ops::MpcSigmoidCrossEntropyWithLogitsOpMaker,
    ops::MpcSigmoidCrossEntropyWithLogitsGradOpMaker<paddle::framework::OpDesc>,
    ops::MpcSigmoidCrossEntropyWithLogitsInplaceInferer);
REGISTER_OPERATOR(mpc_sigmoid_cross_entropy_with_logits_grad,
                  ops::MpcSigmoidCrossEntropyWithLogitsGradOp,
                  ops::MpcSigmoidCrossEntropyWithLogitsGradInplaceInferer);
#ifdef USE_CUDA

REGISTER_OP_CUDA_KERNEL(
    mpc_sigmoid_cross_entropy_with_logits,
    ops::MpcSigmoidCrossEntropyWithLogitsKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    mpc_sigmoid_cross_entropy_with_logits_grad,
    ops::MpcSigmoidCrossEntropyWithLogitsGradKernel<paddle::platform::CUDADeviceContext, int64_t>);

#else // USE_CUDA

REGISTER_OP_CPU_KERNEL(
    mpc_sigmoid_cross_entropy_with_logits,
    ops::MpcSigmoidCrossEntropyWithLogitsKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    mpc_sigmoid_cross_entropy_with_logits_grad,
    ops::MpcSigmoidCrossEntropyWithLogitsGradKernel<paddle::platform::CPUDeviceContext, int64_t>);

#endif // USE_CUDA
