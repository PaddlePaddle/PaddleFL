/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "./conv_op.h"

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

std::vector<int64_t> ConvOp::ComputeOutputShape(
    framework::InferShapeContext* ctx) const {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Conv");
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "Conv");

    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");

    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::string padding_algorithm =
        ctx->Attrs().Get<std::string>("padding_algorithm");
    int groups = ctx->Attrs().Get<int>("groups");
    std::vector<int> dilations = ctx->Attrs().Get<std::vector<int>>("dilations");
    const std::string data_format = ctx->Attrs().Get<std::string>("data_format");

    // MKL-DNN Kernels are using NCHW order of dims description
    // so we ignore data_format consideration for MKL-DNN kernel
    const bool channel_last = (this->IsMKLDNNType() == false) &&
        (data_format == "NHWC" || data_format == "NDHWC");

    PADDLE_ENFORCE_EQ(
        // 1 for share dim
        in_dims.size() == 4 + 1 || in_dims.size() == 5 + 1, true,
        platform::errors::InvalidArgument(
            "The input of Op(Conv) should be a 4-D or 5-D Tensor. But "
            "received: input's dimension is %u, input's shape is [%s].",
            in_dims.size(), in_dims));

    PADDLE_ENFORCE_EQ(
        in_dims.size(), filter_dims.size(),
        platform::errors::InvalidArgument(
            "The input's dimension and filter's dimension of "
            "Op(Conv) should be equal. But received: the input's shape is [%s], "
            "the input's dimension is %d; the filter's shape is [%s],  "
            "the filter's dimension is %d.",
            in_dims, in_dims.size(), filter_dims, filter_dims.size()));

    int in_sub_stride_size = in_dims.size() - strides.size();
    PADDLE_ENFORCE_EQ(
        in_dims.size(), strides.size() + 2U + 1,
        platform::errors::InvalidArgument(
            "The difference of input's dimension and Attr(strides)'s "
            "length must be euqal to 2 for Op(Conv). "
            "But received: input's dimension is %d, input's shape is [%s]; "
            "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
            "difference of input's dimention and Attr(strides)'s length = %u.",
            in_dims.size(), in_dims, strides.size(),
            framework::make_ddim(strides), in_sub_stride_size));

    const auto input_channels =
        channel_last ? in_dims[in_dims.size() - 1] : in_dims[1 + 1];

    PADDLE_ENFORCE_EQ(
        input_channels, filter_dims[1 + 1] * groups,
        platform::errors::InvalidArgument(
            "The number of input's channels should be equal to filter's channels "
            "* groups for Op(Conv). But received: the input's channels is %d, "
            "the input's shape is [%s]; the filter's channels is %d, the "
            "filter's shape is [%s]; the groups is %d, the data_format is %s. "
            "The error may come from wrong data_format setting.",
            input_channels, in_dims, filter_dims[1 + 1], filter_dims, groups,
            data_format));
    PADDLE_ENFORCE_EQ(
        filter_dims[0 + 1] % groups, 0,
        platform::errors::InvalidArgument(
            "The number of output's channels (filter's first dimension) of "
            "Op(Conv) should be divided by groups. But received: "
            "the output channels is %d, the filter's shape is [%s], "
            "the groups is %d.",
            filter_dims[0 + 1], filter_dims, groups));

    framework::DDim in_data_dims;
    if (channel_last) {
        in_data_dims = framework::slice_ddim(in_dims, 1 + 1, in_dims.size() - 1);
    } else {
        in_data_dims = framework::slice_ddim(in_dims, 2 + 1, in_dims.size());
    }

    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2 + 1, filter_dims.size());

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    std::vector<int64_t> output_shape({in_dims[0], in_dims[1]});
    if (!channel_last) {
        output_shape.push_back(filter_dims[0 + 1]);
    }
    for (int i = 0; i < in_data_dims.size(); ++i) {
        if ((!ctx->IsRuntime()) &&
            (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
            output_shape.push_back(-1);
        } else {
            output_shape.push_back(
                ConvOutputSize(in_data_dims[i], filter_data_dims[i], dilations[i],
                               paddings[2 * i], paddings[2 * i + 1], strides[i]));
        }
    }
    if (channel_last) {
        output_shape.push_back(filter_dims[1]);
    }

    return output_shape;
}

framework::OpKernelType ConvOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    framework::LibraryType library{framework::LibraryType::kPlain};
    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    std::string data_format =
        "AnyLayout";  // todo enable data layout when it's ready
    framework::DataLayout layout = framework::StringToDataLayout(data_format);

    if (input_data_type != framework::proto::VarType::INT8 &&
        input_data_type != framework::proto::VarType::UINT8) {
        auto filter_data_type = ctx.Input<Tensor>("Filter")->type();
        PADDLE_ENFORCE_EQ(input_data_type, filter_data_type,
                          platform::errors::InvalidArgument(
                              "input and filter data type should be consistent"));
    }
    if (input_data_type == framework::proto::VarType::FP16) {
        PADDLE_ENFORCE_EQ(library, framework::LibraryType::kCUDNN,
                          platform::errors::InvalidArgument(
                              "float16 can only be used when CUDNN is used"));
    }

    auto type = framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                        library, customized_type_value);
    return type;
}

framework::OpKernelType ConvOp::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
}

void Conv2DOpMaker::Make() {
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddInput("Input",
             "(Tensor) The input tensor of convolution operator. "
             "The format of input tensor is NCHW or NHWC, where N is batch size, "
             "C is the "
             "number of channels, H is the height of the feature, "
             "and W is the width of the feature.");
    AddInput("Filter",
             "(Tensor) The filter tensor of convolution operator. "
             "The format of the filter tensor is MCHW, where M is the number of "
             "output image channels, C is the number of input image channels, "
             "H is the height of the filter, and W is the width of the filter. "
             "If the groups attribute is greater than 1, C equals the number of "
             "input image channels divided by the groups.");
    AddInput("Bias",
             "(Tensor) Bias to be added to each output of filter application."
             "The format of output tensor is X (one-dimensional) of size equal"
             "to the number of output channels. Only used with MKL-DNN.")
        .AsDispensable();
    AddOutput("Output",
              "(Tensor) The output tensor of convolution operator. "
              "It has same data fromat and data type as the Input.");
    AddAttr<std::vector<int>>("strides",
                              "(vector<int> default:{1, 1}), the "
                              "strides(h_stride, w_stride) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int> default:{0, 0}), the "
                              "paddings(pad_height_top, pad_height_bottom, "
                              "pad_width_left, pad_wifth_right)  of "
                              "convolution operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>(
        "padding_algorithm",
        "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
        "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
        "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
        .SetDefault("EXPLICIT");
    AddAttr<int>(
        "groups",
        "(int default:1), the groups number of the convolution operator. "
        "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
        "when group=2, the first half of the filters is only connected to the "
        "first half of the input channels, while the second half of the filters "
        "is only connected to the second half of the input channels.")
        .SetDefault(1);
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int> default:{1, 1}), the "
                              "dilations(h_dilation, w_dilation) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<bool>("use_quantizer",
                  "(bool, default false) "
                  "Set to true for operators that should be quantized and use "
                  "int8 kernel. "
                  "Only used on CPU.")
        .SetDefault(false);
    AddAttr<float>("Scale_in",
                   "Scale_in to be used for int8 input data."
                   "Only used with MKL-DNN INT8.")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_out",
                   "Scale_out to be used for int8 output data."
                   "Only used with MKL-DNN INT8.")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_in_eltwise",
                   "Scale_in_eltwise to be used for int8 eltwise input data."
                   "Only used with MKL-DNN INT8.")
        .SetDefault(1.0f);
    AddAttr<std::vector<float>>("Scale_weights",
                                "Scale_weights to be used for int8 weights data."
                                "Only used with MKL-DNN INT8.")
        .SetDefault({1.0f});
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("NCHW");
    // TODO(dzhwinter): need to registered layout transform function
    AddAttr<bool>("exhaustive_search",
                  "(bool, default false) cuDNN has many algorithm to calculation "
                  "convolution, whether enable exhaustive search "
                  "for cuDNN convolution or not, default is False.")
        .SetDefault(false);

    AddComment(R"DOC(
Convolution Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, dilations, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and Output(Output) are in NCHW or NHWC format. Where N is batch
size, C is the number of channels, H is the height of the feature, and W is
the width of the feature.
Filters(Input) is MCHW format format. Where M is the number of output image channels, C is
the number of input image channels, H is the height of the filter, and W
is the width of the filter.
Parameters(strides, paddings, dilations) are two elements. These two elements represent
height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, H_f, W_f)$
  Output:
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
  Where
$$
       H_{out}= \frac{(H_{in} + pad_height_top + pad_height_bottom - (dilations[0] * (H_f - 1) + 1))}{strides[0]}+ 1 \\
       W_{out}= \frac{(W_{in} + pad_width_left + pad_width_right - (dilations[1] * (W_f - 1) + 1))}{strides[1]}+ 1
$$
)DOC");
        Apply();
}

void ConvOpGrad::InferShape(framework::InferShapeContext* ctx) const {
    auto in_dims = ctx->GetInputDim("Input");
    auto filter_dims = ctx->GetInputDim("Filter");
    if (ctx->HasOutput(framework::GradVarName("Input"))) {
        ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
        ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
    }
}

framework::OpKernelType ConvOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    framework::LibraryType library_{framework::LibraryType::kPlain};
    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    std::string data_format = "AnyLayout";
    framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

    auto type = framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"), ctx.GetPlace(),
        layout_, library_, customized_type_value);
    return type;
}

framework::OpKernelType ConvOpGrad::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
}

template <typename T>
class Conv2DGradMaker : public framework::SingleGradOpMaker<T> {
public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

    void Apply(GradOpPtr<T> op) const override {
        op->SetType(this->ForwardOpType() + "_grad");
        op->SetInput("Input", this->Input("Input"));
        op->SetInput("Filter", this->Input("Filter"));
        op->SetInput("Bias", this->Input("Bias"));
        op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

        op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
        op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));
        op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
        op->SetAttrMap(this->Attrs());
    }
};

template <typename T>
struct CopyData<platform::CPUDeviceContext, T> {
    void operator()(T* dst, const T* src, size_t numel) {
        std::memcpy(dst, src, sizeof(T) * numel);
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mpc_conv2d, ops::ConvOp, ops::Conv2DOpMaker,
                  ops::ConvOpInferVarType,
                  ops::Conv2DGradMaker<paddle::framework::OpDesc>,
                  ops::Conv2DGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(mpc_conv2d_grad, ops::ConvOpGrad);

REGISTER_OP_CPU_KERNEL(
    mpc_conv2d, ops::GemmConvKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    mpc_conv2d_grad,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
