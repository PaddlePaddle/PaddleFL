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
#include "paddle/fluid/framework/data_layout.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "mpc_batch_norm_op.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class MpcBatchNormOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override{
        OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchNorm");
        OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "BatchNorm");
        OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "BatchNorm");
        OP_INOUT_CHECK(ctx->HasInput("Mean"), "Input", "Mean", "BatchNorm");
        OP_INOUT_CHECK(ctx->HasInput("Variance"), "Input", "Variance", "BatchNorm");
        OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "BatchNorm");

        bool is_test = ctx->Attrs().Get<bool>("is_test");
        bool trainable_stats = ctx->Attrs().Get<bool>("trainable_statistics");
        bool test_mode = is_test && (!trainable_stats);
        if (!test_mode) {
            OP_INOUT_CHECK(ctx->HasOutput("MeanOut"), "Output", "MeanOut", "BatchNorm");
            OP_INOUT_CHECK(ctx->HasOutput("VarianceOut"), "Output", "VarianceOut",
                           "BatchNorm");
            OP_INOUT_CHECK(ctx->HasOutput("SavedMean"), "Output", "SavedMean",
                           "BatchNorm");
            OP_INOUT_CHECK(ctx->HasOutput("SavedVariance"), "Output", "SavedVariance",
                           "BatchNorm");
        }

        // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
        PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                          platform::errors::InvalidArgument(
                              "Mean and MeanOut should share the same memory"));

        PADDLE_ENFORCE_EQ(
            ctx->Inputs("Variance")[0], ctx->Outputs("VarianceOut")[0],
            platform::errors::InvalidArgument(
                "Variance and VarianceOut should share the same memory"));

        const auto x_dims = ctx->GetInputDim("X");
        const DataLayout data_layout = framework::StringToDataLayout(
            ctx->Attrs().Get<std::string>("data_layout"));

        if (ctx->IsRuntime() && ctx->HasInput("MomentumTensor")) {
            auto mom = ctx->Inputs("MomentumTensor");
            PADDLE_ENFORCE_EQ(mom.size(), 1,
                              platform::errors::InvalidArgument(
                                  "The input tensor MomentumTensor's size must be 1"
                                  "But received: MomentumTensor's size is [%d]",
                                  mom.size()));
        }

        PADDLE_ENFORCE_GE(
            x_dims.size(), 3,
            platform::errors::InvalidArgument(
                "ShapeError: the dimension of input "
                "X must greater than or equal to 3. But received: the shape of input "
                "X = [%s], the dimension of input X =[%d]",
                x_dims, x_dims.size()));

        PADDLE_ENFORCE_LE(
            x_dims.size(), 6,
            platform::errors::InvalidArgument(
                "ShapeError: the dimension of input X "
                "must smaller than or equal to 6. But received: the shape of input X "
                "= [%s], the dimension of input X = [%d]",
                x_dims, x_dims.size()));


        const int64_t C =
            ((this->IsMKLDNNType() == true) || (data_layout == DataLayout::kNCHW)
                 ? x_dims[2]
                 : x_dims[x_dims.size() - 1]);

        auto scale_dim = ctx->GetInputDim("Scale");
        auto bias_dim = ctx->GetInputDim("Bias");
        VLOG(3) << "*** scale_dims: " << scale_dim;
        VLOG(3) << "*** bias_dims: " << bias_dim;
        VLOG(3) << "*** mean_dims: " << ctx->GetInputDim("Mean");
        VLOG(3) << "*** variance_dims: " << ctx->GetInputDim("Variance");
        //VLOG(3) << "*** Y_dims: " << ctx->GetInputDim("Y");

        PADDLE_ENFORCE_EQ(
            scale_dim.size(), 2UL,
            platform::errors::InvalidArgument(
                "ShapeError: the dimension of scale must equal to 2."
                "But received: the shape of scale is [%s], the dimension "
                "of scale is [%d]",
                scale_dim, scale_dim.size()));
        PADDLE_ENFORCE_EQ(bias_dim.size(), 2UL,
            platform::errors::InvalidArgument(
                "ShapeError: the dimension of bias must equal to 2."
                "But received: the shape of bias is [%s],the dimension "
                "of bias is [%d]",
                bias_dim, bias_dim.size()));

        bool check = true;
        if ((!ctx->IsRuntime()) && (framework::product(scale_dim) <= 0 ||
                                    framework::product(bias_dim) <= 0)) {
            check = false;
        }

        if (check) {
            PADDLE_ENFORCE_EQ(scale_dim[1], C,
                platform::errors::InvalidArgument(
                    "ShapeError: the shape of scale must equal to [%d]"
                    "But received: the shape of scale is [%d]",
                    C, scale_dim[1]));
            PADDLE_ENFORCE_EQ(bias_dim[1], C,
                platform::errors::InvalidArgument(
                    "ShapeError: the shape of bias must equal to [%d]"
                    "But received: the shape of bias is [%d]",
                    C, bias_dim[1]));
        }
        ctx->SetOutputDim("Y", x_dims);
        ctx->SetOutputDim("MeanOut", {2, C}); // 2: share_num
        ctx->SetOutputDim("VarianceOut", {2, C});
        ctx->SetOutputDim("SavedMean", {2, C});
        ctx->SetOutputDim("SavedVariance", {2, C});
        ctx->ShareLoD("X", "Y");
  }

protected:
    framework::OpKernelType GetExpectedKernelType(const framework::ExecutionContext& ctx) const {
        framework::LibraryType library_{framework::LibraryType::kPlain};
        std::string data_format = "AnyLayout";
        framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

        return framework::OpKernelType(
            OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
            layout_, library_);
    }

    framework::OpKernelType GetKernelTypeForVar(
            const std::string& var_name, const Tensor& tensor,
            const framework::OpKernelType& expected_kernel_type) const {
        return framework::OpKernelType(expected_kernel_type.data_type_,
                                       tensor.place(), tensor.layout());
    }
};


class MpcBatchNormGradOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override{
        // check input
        OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "BatchNormGrad");
        OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                       framework::GradVarName("Y"), "BatchNormGrad");
        OP_INOUT_CHECK(ctx->HasInput("SavedMean"), "Input", "SavedMean",
                       "BatchNormGrad");
        OP_INOUT_CHECK(ctx->HasInput("SavedVariance"), "Input", "SavedVariance",
                       "BatchNormGrad");

        // check output
        OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                       framework::GradVarName("X"), "BatchNormGrad");

        const bool has_scale_grad = ctx->HasOutput(framework::GradVarName("Scale"));
        const bool has_bias_grad = ctx->HasOutput(framework::GradVarName("Bias"));

        PADDLE_ENFORCE_EQ((has_scale_grad == has_bias_grad), true,
                          platform::errors::NotFound(
                              "Output(Scale@GRAD) and Output(Bias@GRAD) must be null "
                              "or not be null at same time. But now, "
                              "has Scale@Grad=[%d], has Bias@GRAD=[%d]",
                              has_scale_grad, has_bias_grad));

        const bool use_global_stats = ctx->Attrs().Get<bool>("use_global_stats");
        if (use_global_stats) {
            PADDLE_ENFORCE_EQ(
                !ctx->Attrs().Get<bool>("use_mkldnn"), true,
                platform::errors::InvalidArgument(
                    "Using global stats during training is not supported "
                    "in gradient op kernel of batch_norm_mkldnn_op now."));
        }

        OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchNormGrad");
        const auto x_dims = ctx->GetInputDim("X");
        const DataLayout data_layout = framework::StringToDataLayout(
            ctx->Attrs().Get<std::string>("data_layout"));

        const int C =
            ((this->IsMKLDNNType() == true) || (data_layout == DataLayout::kNCHW)
                ? x_dims[2]
                : x_dims[x_dims.size() - 1]);

        ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
        // has_scale_grad == has_bias_grad, judge has_scale_grad is enough
        if (has_scale_grad) {
            ctx->SetOutputDim(framework::GradVarName("Scale"), {2, C}); // 2: share_num
            ctx->SetOutputDim(framework::GradVarName("Bias"), {2, C});
        }
    }

protected:
    framework::OpKernelType GetExpectedKernelType(const framework::ExecutionContext& ctx) const {
        framework::LibraryType library_{framework::LibraryType::kPlain};
        std::string data_format = "AnyLayout";
        framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

        auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
        return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout_, library_);
    }

    framework::OpKernelType GetKernelTypeForVar(
            const std::string& var_name, const Tensor& tensor,
            const framework::OpKernelType& expected_kernel_type) const {
        return framework::OpKernelType(expected_kernel_type.data_type_,
                                       tensor.place(), tensor.layout());
    }
};


class MpcBatchNormOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_GE(
            epsilon, 0.0f,
            platform::errors::InvalidArgument(
                "'epsilon' should be greater or equal than 0.0."));
        PADDLE_ENFORCE_LE(epsilon, 0.001f,
                          platform::errors::InvalidArgument(
                              "'epsilon' should be less or equal than 0.001."));
      });
  AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
  AddInput("X", "The input tensor");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Mean",
           "The global mean (for training) or "
           "estimated mean (for testing)");
  AddInput("Variance",
           "The global variance (for training) "
           "or estimated Variance (for testing)");
  AddInput("MomentumTensor",
           "(Tensor<float32>, optional) If provided, batch_norm will "
           "use this as momentum, this has a higher priority than "
           "attr(momentum), the shape of this tensor MUST BE [1].")
      .AsDispensable();
  AddOutput("Y", "result after normalization");
  AddOutput("MeanOut",
            "Share memory with Mean. "
            "Store the global mean when training");
  AddOutput("VarianceOut",
            "Share memory with Variance. "
            "Store the global Variance when training");
  AddOutput("SavedMean",
            "Mean of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("SavedVariance",
            "Variance of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("ReserveSpace",
            "Reserve GPU space for triggering the new semi-persistent "
            "NHWC kernel")
      .AsDispensable();
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_with_relu",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("use_global_stats",
                "(bool, default false) Whether to use global mean and "
                "variance. In inference or test mode, set use_global_stats "
                "to true or is_test true. the behavior is equivalent. "
                "In train mode, when setting use_global_stats True, the "
                "global mean and variance are also used during train time, "
                "the BN acts as scaling and shiffting.")
      .SetDefault(false);
  AddAttr<bool>("trainable_statistics",
                "(bool, default false) Whether to calculate mean and variance "
                "in test mode. If setting true in test mode, mean and variace "
                "will be calculated by current batch statistics.")
      .SetDefault(false);
  AddComment(R"DOC(
Batch Normalization.
Batch Norm has been implemented as discussed in the paper:
https://arxiv.org/pdf/1502.03167.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is one of the following:
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`
)DOC");
    }
};

template <typename T>
class MpcBatchNormGradOpMaker : public framework::SingleGradOpMaker<T> {
public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
    void Apply(GradOpPtr<T> op) const {
        op->SetType(this->ForwardOpType() + "_grad");
        op->SetInput("X", this->Input("X"));
        op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

        op->SetInput("Scale", this->Input("Scale"));
        op->SetInput("Bias", this->Input("Bias"));
        op->SetInput("SavedMean", this->Output("SavedMean"));
        op->SetInput("SavedVariance", this->Output("SavedVariance"));
        if (this->HasOutput("ReserveSpace")) {
            op->SetInput("ReserveSpace", this->Output("ReserveSpace"));
        }

        // used when setting use_global_stats True during training
        if (boost::get<bool>(this->GetAttr("use_global_stats"))) {
            op->SetInput("Mean", this->Output("MeanOut"));
            op->SetInput("Variance", this->Output("VarianceOut"));
        }
        op->SetAttrMap(this->Attrs());
        op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
        op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
        op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    }
};


class MpcBatchNormOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
protected:
    std::unordered_map<std::string, std::string>& GetInputOutputWithSameType() const override {
        static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
        return m;
    }
};

template <typename T>
struct Expand<platform::CPUDeviceContext, T> {

    void operator()(const Tensor* input, Tensor* output, int S, int N, int C, int sample_size) {
        // Expand tensor into specified shape
        // input shape: {S, C}
        // outout shape: {S, N, C, H, W}, sample_size = H * W
        const T* input_data = input->data<T>();
        T* output_data = output->data<T>();
        int input_share_offset = C;
        int output_share_offset = N * C * sample_size;
        for (int nc = 0; nc < N * C; ++nc) {
            int nc_offset = nc * sample_size;
            std::fill(output_data + nc_offset, output_data + nc_offset + sample_size, *(input_data + nc % C));
            std::fill(output_data + nc_offset + output_share_offset,
                      output_data + nc_offset + output_share_offset + sample_size,
                      *(input_data + nc % C + input_share_offset));
        }
    }

};

std::shared_ptr<mpc::MpcOperators> mpc_operators;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    mpc_batch_norm, ops::MpcBatchNormOp, ops::MpcBatchNormOpMaker,
    ops::MpcBatchNormOpInferVarType,
    ops::MpcBatchNormGradOpMaker<paddle::framework::OpDesc>,
    ops::MpcBatchNormGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mpc_batch_norm_grad, ops::MpcBatchNormGradOp);

#ifndef USE_CUDA
REGISTER_OP_CPU_KERNEL(
    mpc_batch_norm, ops::MpcBatchNormKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    mpc_batch_norm_grad, ops::MpcBatchNormGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
#endif // USE_CUDA
