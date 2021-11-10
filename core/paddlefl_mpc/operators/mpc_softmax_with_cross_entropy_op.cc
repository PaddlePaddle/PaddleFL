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
#include "mpc_softmax_with_cross_entropy_op.h"

namespace paddle {
namespace operators {

class MpcSoftmaxWithCrossEntropyOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("Logits"), true,
            platform::errors::InvalidArgument("Input(Logits) should be not null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("Label"), true,
            platform::errors::InvalidArgument("Input(Label) should be not null."));

        PADDLE_ENFORCE_EQ(ctx->HasOutput("Softmax"), true,
                          platform::errors::InvalidArgument(
                              "Output(Softmax) should be not null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasOutput("Loss"), true,
            platform::errors::InvalidArgument("Output(Loss) should be not null."));

        auto axis = ctx->Attrs().Get<int>("axis");
        auto logits_dims = ctx->GetInputDim("Logits");
        auto labels_dims = ctx->GetInputDim("Label");
        auto logits_rank = logits_dims.size();

        axis = CanonicalAxis(axis, logits_rank);
        PADDLE_ENFORCE_GE(axis, logits_rank - 1,
                          platform::errors::InvalidArgument(
                              "Attr(axis) value should be -1 or R-1, "
                              "R is the rank of Input(Logits)."));
        for (int i = 0; i < logits_rank; i++) {
            if (i != axis) {
                if (ctx->IsRuntime() || (logits_dims[i] > 0 && labels_dims[i] > 0)) {
                    PADDLE_ENFORCE_EQ(logits_dims[i], labels_dims[i],
                                      platform::errors::InvalidArgument(
                                          "Input(Logits) and Input(Label) should in "
                                          "same shape in dimensions except axis."));
                }
            }
        }

        bool soft_label = ctx->Attrs().Get<bool>("soft_label");
        PADDLE_ENFORCE_EQ(soft_label, true,
                          platform::errors::InvalidArgument(
                              "soft_label can only be true! "));
        if (soft_label) {
            if (ctx->IsRuntime() ||
                    (logits_dims[axis] > 0 && labels_dims[axis] > 0)) {
                PADDLE_ENFORCE_EQ(logits_dims[axis], labels_dims[axis],
                                  platform::errors::InvalidArgument(
                                      "If Attr(soft_label) == true,  "
                                      "the axis dimension of "
                                      "Input(X) and Input(Label) should be equal."));
            }
        }
        ctx->SetOutputDim("Softmax", logits_dims);

        logits_dims[axis] = 1;
        ctx->SetOutputDim("Loss", logits_dims);

        ctx->ShareLoD("Logits", /*->*/ "Softmax");
        ctx->ShareLoD("Logits", /*->*/ "Loss");
    }
};


class MpcSoftmaxWithCrossEntropyOpGrad : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Loss")), true,
                          platform::errors::InvalidArgument(
                              "Input(Loss@Grad) should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasInput("Softmax"), true,
                          platform::errors::InvalidArgument(
                              "Input(Softmax) should be not null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("Label"), true,
            platform::errors::InvalidArgument("Input(Label) should be not null."));
        PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Logits")), true,
                          platform::errors::InvalidArgument(
                              "Output(Logits@Grad) should be not null."));

        auto axis = ctx->Attrs().Get<int>("axis");
        auto softmax_dims = ctx->GetInputDim("Softmax");
        auto labels_dims = ctx->GetInputDim("Label");
        auto softmax_rank = softmax_dims.size();

        axis = CanonicalAxis(axis, softmax_rank);
        PADDLE_ENFORCE_GE(axis, softmax_rank - 1,
                          platform::errors::InvalidArgument(
                              "Attr(axis) value should be -1 or R-1, "
                              "R is the rank of Input(Logits)."));
        for (int i = 0; i < softmax_rank; i++) {
            if (i != axis) {
                if (ctx->IsRuntime() || (softmax_dims[i] > 0 && labels_dims[i] > 0)) {
                    PADDLE_ENFORCE_EQ(
                        softmax_dims[i], labels_dims[i],
                        platform::errors::InvalidArgument(
                            "Input(Logits) and Input(Label) should in same shape in "
                            "dimensions except axis."));
                }
            }
        }

        bool soft_label = ctx->Attrs().Get<bool>("soft_label");
        PADDLE_ENFORCE_EQ(soft_label, true,
                          platform::errors::InvalidArgument(
                              "soft_label can only be true! "));
        if (soft_label) {
            if (ctx->IsRuntime() || (softmax_dims[axis] > 0 && labels_dims[axis] > 0)) {
                PADDLE_ENFORCE_EQ(softmax_dims[axis], labels_dims[axis],
                                  platform::errors::InvalidArgument(
                                      "If Attr(soft_label) == true, "
                                      "the axis dimension of "
                                      "Input(X) and Input(Label) should be equal."));
            }
        }

        ctx->SetOutputDim(framework::GradVarName("Logits"),
                          ctx->GetInputDim("Softmax"));
    }
};


class MpcSoftmaxWithCrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("Logits",
                 "(Tensor, default: Tensor<float>), The input tensor of unscaled "
                 "log probabilities, whose dimension :attr:`axis` should be scaled "
                 "by softmax.");
        AddInput(
            "Label",
            "(Tensor) The input tensor of groud truth label. If :attr:`soft_label` "
            "is set to false, Label is a Tensor<int64> in same shape with "
            "Input(Logits) except the shape in dimension :attr:`axis` as 1. If "
            "soft_label is set to true, Label is a Tensor<float/double> in same "
            "shape with Input(Logits).");
        AddOutput(
            "Softmax",
            "(Tensor, default: Tensor<float>), A tensor in same shape with "
            "Input(Logits). "
            "The outputs value of softmax activation by given the input batch, "
            "which will be used in backward calculation.")
            .AsIntermediate();
        AddOutput("Loss",
                  "(Tensor, default: Tensor<float>), A tensor in same shape with "
                  "Input(Logits) "
                  "except the shape in dimension :attr:`axis` as 1. The cross "
                  "entropy loss.");
        AddAttr<bool>(
            "soft_label",
            "(bool, default: false), A flag to indicate whether to interpretant "
            "the given labels as soft labels.")
            .SetDefault(false);
        AddAttr<int>("axis",
                     "The dimension index of Input(Logits) to perform softmax,"
                     "default -1 for last dimension")
            .SetDefault(-1);
        AddAttr<bool>("use_relu", "").SetDefault(false);
        AddAttr<bool>("use_long_div", "").SetDefault(true);
        AddComment(R"DOC(
Softmax With Cross Entropy Operator.
Cross entropy loss with softmax is used as the output layer extensively. This
operator computes the softmax normalized values for each row of the input
tensor.
Conputing cross-entropy loss is not supported now.
Now, we only support soft_label=true, axis=-1 or (rank-1).
Forward: out = softmax(x). todo: add cross_entropy
backward: dx = dout.expand * (softmax(x) - label)
)DOC");
    }
};


template <typename T>
class MpcSoftmaxGradMaker : public framework::SingleGradOpMaker<T> {
public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
    void Apply(GradOpPtr<T> grad_op) const override {
        grad_op->SetType("mpc_softmax_with_cross_entropy_grad");
        grad_op->SetInput("Label", this->Input("Label"));
        grad_op->SetInput("Softmax", this->Output("Softmax"));
        grad_op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
        grad_op->SetOutput(framework::GradVarName("Logits"),
                           this->InputGrad("Logits"));
        grad_op->SetAttrMap(this->Attrs());
    }
};


DECLARE_INPLACE_OP_INFERER(MpcSoftmaxWithCrossEntropyInplaceInference,
                           {"Logits", "Softmax"});

DECLARE_INPLACE_OP_INFERER(MpcSoftmaxWithCrossEntropyGradInplaceInference,
                           {"Softmax", framework::GradVarName("Logits")});

template <typename T>
struct SetExpandData<paddle::platform::CPUDeviceContext, T> {
    void operator()(T* dst, const T* src, size_t n, size_t d) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                dst[i * d + j] = src[i];
            }
        }
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(mpc_softmax_with_cross_entropy, ops::MpcSoftmaxWithCrossEntropyOp,
                  ops::MpcSoftmaxWithCrossEntropyOpMaker,
                  ops::MpcSoftmaxGradMaker<paddle::framework::OpDesc>,
                  ops::MpcSoftmaxGradMaker<paddle::imperative::OpBase>,
                  ops::MpcSoftmaxWithCrossEntropyInplaceInference);
REGISTER_OPERATOR(mpc_softmax_with_cross_entropy_grad,
                  ops::MpcSoftmaxWithCrossEntropyOpGrad,
                  ops::MpcSoftmaxWithCrossEntropyGradInplaceInference);

#ifndef USE_CUDA

REGISTER_OP_CPU_KERNEL(mpc_softmax_with_cross_entropy,
                       ops::MpcSoftmaxWithCrossEntropyKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(mpc_softmax_with_cross_entropy_grad,
                       ops::MpcSoftmaxWithCrossEntropyGradKernel<paddle::platform::CPUDeviceContext, int64_t>);

#endif // USE_CUDA
