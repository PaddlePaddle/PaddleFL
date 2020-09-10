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

#include "precision_recall_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include <string>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcPrecisionRecallOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Predicts"), true,
                      platform::errors::InvalidArgument(
                          "Input(Predicts) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Labels"), true,
        platform::errors::InvalidArgument("Input(Labels) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("BatchMetrics"), true,
                      platform::errors::InvalidArgument(
                          "Output(BatchMetrics) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("AccumMetrics"), true,
                      platform::errors::InvalidArgument(
                          "Output(AccumMetrics) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("AccumStatesInfo"), true,
                      platform::errors::InvalidArgument(
                          "Output(AccumStatesInfo) should not be null."));

    int64_t cls_num =
        static_cast<int64_t>(ctx->Attrs().Get<int>("class_number"));

    PADDLE_ENFORCE_EQ(cls_num, 1,
                      platform::errors::InvalidArgument(
                          "Only support predicts/labels for 1"
                          "in binary classification for now."));

    auto preds_dims = ctx->GetInputDim("Predicts");
    auto labels_dims = ctx->GetInputDim("Labels");

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(preds_dims, labels_dims,
                        platform::errors::InvalidArgument(
                            "The dimension of Input(Predicts) and "
                            "Input(Labels) should be the same."
                            "But received (%d) != (%d)",
                            preds_dims, labels_dims));
      PADDLE_ENFORCE_EQ(
          labels_dims.size(), 2,
          platform::errors::InvalidArgument(
              "Only support predicts/labels for 1"
              "in binary classification for now."
              "The dimension of Input(Labels) should be equal to 2 "
              "(1 for shares). But received (%d)",
              labels_dims.size()));
    }
    if (ctx->HasInput("StatesInfo")) {
      auto states_dims = ctx->GetInputDim("StatesInfo");

      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            states_dims, framework::make_ddim({2, 3}),
            platform::errors::InvalidArgument(
                "The shape of Input(StatesInfo) should be [2, 3]."));
      }
    }

    // Layouts of BatchMetrics and AccumMetrics both are:
    // [
    //  precision, recall, F1 score,
    // ]
    ctx->SetOutputDim("BatchMetrics", {3});
    ctx->SetOutputDim("AccumMetrics", {3});
    // Shape of AccumStatesInfo is [3]
    // The layout of each row is:
    // [ TP, FP, FN ]
    ctx->SetOutputDim("AccumStatesInfo", {2, 3});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Predicts"),
        ctx.device_context());
  }
};

class MpcPrecisionRecallOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Predicts",
             "(Tensor, default Tensor<int64_t>) A 1-D tensor with shape N, "
             "where N is the batch size. Each element contains the "
             "corresponding predicts of an instance which computed by the "
             "previous sigmoid operator.");
    AddInput("Labels",
             "(Tensor, default Tensor<int>) A 1-D tensor with shape N, "
             "where N is the batch size. Each element is a label and the "
             "value should be in [0, 1].");
    AddInput("StatesInfo",
             "(Tensor, default Tensor<int>) A 1-D tensor with shape 3. "
             "This input is optional. If provided, current state will be "
             "accumulated to this state and the accumulation state will be "
             "the output state.")
        .AsDispensable();
    AddOutput("BatchMetrics",
              "(Tensor, default Tensor<int64_t>) A 1-D tensor with shape {3}. "
              "This output tensor contains metrics for current batch data. "
              "The layout is [precision, recall, f1 score].");
    AddOutput("AccumMetrics",
              "(Tensor, default Tensor<int64_t>) A 1-D tensor with shape {3}. "
              "This output tensor contains metrics for accumulated data. "
              "The layout is [precision, recall, f1 score].");
    AddOutput("AccumStatesInfo",
              "(Tensor, default Tensor<int64_t>) A 1-D tensor with shape 3. "
              "This output tensor contains "
              "accumulated state variables used to compute metrics. The layout "
              "for each class is [true positives, false positives, "
              "false negatives].");
    AddAttr<int>("class_number", "(int) Number of classes to be evaluated.");
    AddAttr<float>("threshold", "(threshold) Threshold of true predict.");
    AddComment(R"DOC(
Precision Recall Operator.
When given Input(Indices) and Input(Labels), this operator can be used
to compute various metrics including:
1. precision
2. recall
3. f1 score
To compute the above metrics, we need to do statistics for true positives,
false positives and false negatives.
We define state as a 1-D tensor with shape [3]. Each element of a
state contains statistic variables for corresponding class. Layout of each row
is: TP(true positives), FP(false positives), FN(false negatives).
This operator also supports metrics computing for cross-batch situation. To
achieve this, Input(StatesInfo) should be provided. State of current batch
data will be accumulated to Input(StatesInfo) and Output(AccumStatesInfo)
is the accumulation state.
Output(BatchMetrics) is metrics of current batch data while
Output(AccumStatesInfo) is metrics of accumulation data.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    mpc_precision_recall, ops::MpcPrecisionRecallOp, ops::MpcPrecisionRecallOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    mpc_precision_recall,
    ops::MpcPrecisionRecallKernel<paddle::platform::CPUPlace, int64_t>);
