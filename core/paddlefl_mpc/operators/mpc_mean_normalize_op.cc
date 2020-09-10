
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

#include "mpc_mean_normalize_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include <string>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcMeanNormalizationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Min"), true,
                      platform::errors::InvalidArgument(
                          "Input(Min) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Max"), true,
        platform::errors::InvalidArgument("Input(Max) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Mean"), true,
        platform::errors::InvalidArgument("Input(Mean) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("SampleNum"), true,
        platform::errors::InvalidArgument("Input(Sample) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Range"), true,
                      platform::errors::InvalidArgument(
                          "Output(Range) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("MeanOut"), true,
                      platform::errors::InvalidArgument(
                          "Output(Meanor) should not be null."));

    int64_t total_sample_num =
        static_cast<int64_t>(ctx->Attrs().Get<int>("total_sample_num"));

    auto min_dims = ctx->GetInputDim("Min");
    auto max_dims = ctx->GetInputDim("Max");
    auto mean_dims = ctx->GetInputDim("Mean");
    auto sample_num_dims = ctx->GetInputDim("SampleNum");

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(min_dims, max_dims,
                        platform::errors::InvalidArgument(
                            "The dimension of Input(Min) and "
                            "Input(Max) should be the same."
                            "But received (%d) != (%d)",
                            min_dims, max_dims));
      PADDLE_ENFORCE_EQ(min_dims, mean_dims,
                        platform::errors::InvalidArgument(
                            "The dimension of Input(Min) and "
                            "Input(Max) should be the same."
                            "But received (%d) != (%d)",
                            min_dims, mean_dims));
      PADDLE_ENFORCE_EQ(
          min_dims.size(), 3,
          platform::errors::InvalidArgument(
              "The dimension of Input(Min) should be equal to 3 "
              "(share_num, party_num, feature_num). But received (%d)",
              min_dims.size()));

      PADDLE_ENFORCE_EQ(
          sample_num_dims.size(), 2,
          platform::errors::InvalidArgument(
              "The dimension of Input(SampleNum) should be equal to 3 "
              "(share_num, party_num). But received (%d)",
              sample_num_dims.size()));

      PADDLE_ENFORCE_EQ(
          sample_num_dims[1], min_dims[1],
          platform::errors::InvalidArgument(
              "The party num of Input(SampleNum) and Input(Min) "
              "should be equal But received (%d) != (%d)",
              sample_num_dims[1], min_dims[1]));
    }

    ctx->SetOutputDim("Range", {mean_dims[0], mean_dims[2]});
    ctx->SetOutputDim("MeanOut", {mean_dims[0], mean_dims[2]});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Min"),
        ctx.device_context());
  }
};

class MpcMeanNormalizationOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Min",
             "(Tensor, default Tensor<int64_t>) A 2-D tensor with shape [P, N], "
             "where P is the party num and N is the feature num. Each row contains "
             " the local min feature val of N features.");
    AddInput("Max",
             "(Tensor, default Tensor<int64_t>) A 2-D tensor with shape [P, N], "
             "where P is the party num and N is the feature num. Each row contains "
             " the local max feature val of N features.");
    AddInput("Mean",
             "(Tensor, default Tensor<int64_t>) A 2-D tensor with shape [P, N], "
             "where P is the party num and N is the feature num. Each row contains "
             " the local mean feature val of N features.");
    AddInput("SampleNum",
             "(Tensor, default Tensor<int64_t>) A 1-D tensor with shape [P], "
             "where P is the party num. Each element contains "
             "sample num of party_i.");
    AddOutput("Range",
              "(Tensor, default Tensor<int64_t>) A 1-D tensor with shape [N], "
              "where N is the feature num. Each element contains "
              "global range of feature_i.");
    AddOutput("MeanOut",
              "(Tensor, default Tensor<int64_t>) A 1-D tensor with shape [N], "
              "where N is the feature num. Each element contains "
              "global mean of feature_i.");
    AddAttr<int>("total_sample_num", "(int) Sum of sample nums from all party.");
    AddComment(R"DOC(
Mean normalization Operator.
When given Input(Min), Input(Max), Input(Mean) and Input(SampleNum),
this operator can be used to compute global range and mean for further feature
scaling.
Output(Range) is the global range of all features.
Output(MeanOut) is the global mean of all features.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    mpc_mean_normalize, ops::MpcMeanNormalizationOp, ops::MpcMeanNormalizationOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    mpc_mean_normalize,
    ops::MpcMeanNormalizationKernel<paddle::platform::CPUPlace, int64_t>);
