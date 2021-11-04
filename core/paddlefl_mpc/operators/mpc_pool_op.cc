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

#include "mpc_pool_op.h"

#include <unordered_map>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

int PoolOutputSize(int input_size, int filter_size, int padding_1,
                   int padding_2, int stride, bool ceil_mode) {
    int output_size;
    if (!ceil_mode) {
        output_size = (input_size - filter_size + padding_1 + padding_2) / stride + 1;
    } else {
        output_size = (input_size - filter_size + padding_1 + padding_2 + stride - 1) / stride + 1;
    }
    PADDLE_ENFORCE_GT(
        output_size, 0,
        "ShapeError: the output size must be greater than 0. But received: "
        "output_size = %d due to the settings of input_size(%d), padding(%d,%d), "
        "k_size(%d) and stride(%d). Please check again!",
        output_size, input_size, padding_1, padding_2, filter_size, stride);
    return output_size;
}


class MpcPoolOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override{
        PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                          "X(Input) of Pooling should not be null.");
        PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                          "Out(Output) of Pooling should not be null.");

        std::string pooling_type = ctx->Attrs().Get<std::string>("pooling_type");
        std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
        std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
        std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
        bool ceil_mode = ctx->Attrs().Get<bool>("ceil_mode");
        // bool adaptive = ctx->Attrs().Get<bool>("adaptive");
        bool global_pooling = ctx->Attrs().Get<bool>("global_pooling");
        std::string data_format = ctx->Attrs().Get<std::string>("data_format");
        std::string padding_algorithm = ctx->Attrs().Get<std::string>("padding_algorithm");

        auto in_x_dims = ctx->GetInputDim("X");
        PADDLE_ENFORCE_EQ(in_x_dims.size(), 5,
            "ShapeError: the input of Op(pool) should be 5-D Tensor (ciphertext). "
            "But received: %u-D Tensor and it's shape is [%s].",
            in_x_dims.size(), in_x_dims);

        PADDLE_ENFORCE_EQ(in_x_dims.size() - ksize.size(), 3U,
            "ShapeError: the dimension of input(ciphertext) minus the size of "
            "Attr(ksize)(plaintext) must be euqal to 3 in Op(pool). "
            "But received: the dimension of input minus the size "
            "of Attr(ksize) is %d, the "
            "input's dimension is %d, the shape of input "
            "is [%s], the Attr(ksize)'s size is %d, the Attr(ksize) is [%s].",
            in_x_dims.size() - ksize.size(), in_x_dims.size(), in_x_dims,
            ksize.size(), framework::make_ddim(ksize));

        PADDLE_ENFORCE_EQ(ksize.size(), strides.size(),
                          "ShapeError: the size of Attr(ksize) and Attr(strides) in "
                          "Op(pool) must be equal. "
                          "But received: Attr(ksize)'s size is %d, Attr(strides)'s "
                          "size is %d, Attr(ksize) is [%s], Attr(strides)is [%s].",
                          ksize.size(), strides.size(), framework::make_ddim(ksize),
                          framework::make_ddim(strides));

        PADDLE_ENFORCE_EQ(data_format, "NCHW",
            "data format can only be 'NCHW' ",
            in_x_dims.size(), in_x_dims);

        // update paddings if "SAME" or global_pooling
        framework::DDim data_dims;
        data_dims = framework::slice_ddim(in_x_dims, 3, in_x_dims.size());
        UpdatePadding(&paddings, global_pooling, padding_algorithm,
                      data_dims, strides, ksize);

        if (global_pooling) {
            UpdateKsize(&ksize, data_dims);
        }

        std::vector<int64_t> output_shape;
        std::vector<int64_t> one_hot_tensor_shape;
        for (int i = 0; i < data_dims.size(); ++i) {
            if ((!ctx->IsRuntime()) && (data_dims[i] < 0)) {
                output_shape.push_back(data_dims[i]);
            } else {
                output_shape.push_back(
                PoolOutputSize(data_dims[i], ksize[i], paddings[2 * i],
                               paddings[2 * i + 1], strides[i], ceil_mode));
            }
        }

        output_shape.insert(output_shape.begin(), in_x_dims[0]); // share size
        output_shape.insert(output_shape.begin() + 1, in_x_dims[1]); // output_N = input_N
        output_shape.insert(output_shape.begin() + 2, in_x_dims[2]); // output_C = input_C

        one_hot_tensor_shape.push_back(in_x_dims[0]); // share size
        one_hot_tensor_shape.push_back(in_x_dims[1]); // input_N
        one_hot_tensor_shape.push_back(in_x_dims[2]); // input_C
        one_hot_tensor_shape.push_back(ksize[0] * ksize[1]);
        one_hot_tensor_shape.push_back(output_shape[3] * output_shape[4]);

        ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
        ctx->ShareLoD("X", "Out");

        ctx->SetOutputDim("One_hot_tensor", framework::make_ddim(one_hot_tensor_shape));
        ctx->ShareLoD("X", "One_hot_tensor");
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


class MpcPoolOpGrad : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override{
        PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) must not be null.");
        PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                          "Input(X@GRAD) should not be null.");
        ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
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


class MpcPool2dOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override{
        AddInput("X",
            "(Tensor) The input tensor of pooling operator. "
            "The format of input tensor is NCHW, where N is batch size, C is the "
            "number of channels, H is the height of the feature, "
            "and W is the width of the feature.");
        AddOutput("Out",
            "(Tensor) The output tensor of pooling operator. "
            "The format of output tensor is also NCHW, "
            "where N is batch size, C is the number of channels, "
            "H is the height of the feature, "
            "and W is the width of the feature.");
        AddOutput("One_hot_tensor",
            "one hot tensor");
        AddAttr<std::string>("pooling_type",
            "(string), pooling type, can be \"max\" for max-pooling "
            "and \"avg\" for average-pooling.")
            .InEnum({"max", "avg"});
        AddAttr<std::vector<int>>("ksize",
            "(vector<int>) The pooling window "
            "size(height, width) of the pooling operator. "
            "If global_pooling = true, ksize and paddings will "
            "be ignored.");
        AddAttr<bool>("global_pooling",
            "(bool) Whether to use the global pooling. "
            "If global_pooling = true, kernel size and paddings will be ignored. "
            "Default False.")
            .SetDefault(false);
        AddAttr<std::vector<int>>("strides",
             "(vector<int>, default {1, 1}), strides(height, "
             "width) of pooling operator.")
            .SetDefault({1, 1});
        AddAttr<std::vector<int>>("paddings",
            "(vector<int>, default {0,0}), paddings(height_top, height_bottom, "
            "width_left, wifth_right) of pooling operator."
            "If global_pooling = true, paddings and kernel size will be ignored.")
            .SetDefault({0, 0});
        AddAttr<bool>("exclusive",
            "(bool) When true, will exclude the zero-padding in the "
            "averaging calculating, otherwise, include the zero-padding. Note, it "
            "is only used when pooling_type is avg. The default is True. "
            "Default True.")
            .SetDefault(true);
        AddAttr<bool>("ceil_mode",
            "(bool) Whether to use the ceil function to calculate "
            "output height and width. False is the default. If it is set to False, "
            "the floor function will be used. Default False")
            .SetDefault(false);
        AddAttr<std::string>("data_format",
            "(string, default NCHW) Only used in "
            "An optional string from: \"NHWC\", \"NCHW\". "
            "Defaults to \"NHWC\". Specify the data format of the output data, "
            "the input will be transformed automatically. ")
            .SetDefault("NCHW");
        AddAttr<bool>("is_test",
            "(bool, default false) Set to true for inference only, false "
            "for training. Some layers may run faster when this is true.")
            .SetDefault(false);
        AddAttr<std::string>("padding_algorithm",
            "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
            "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
            "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
           .SetDefault("EXPLICIT");
        AddComment(R"DOC(
This operation calculates the pooling output based on
the input, pooling_type and pool_size, pool_stride, pool_padding parameters.
Input(X) and Output(Out) are in NCHW or NHWC format, where N is batch size, C is the
number of channels, H is the height of the feature, and W is the width of the feature.
Parameters(pool_size, pool_stride, pool_padding) hold two integer elements.
These two elements represent height and width, respectively.
The input(X) size and output(Out) size may be different.
)DOC");
    }
};

class MpcPoolOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
protected:
    std::unordered_map<std::string, std::string>& GetInputOutputWithSameType() const override {
        static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
        return m;
    }
};

template <typename T, typename Func>
struct VisitDataStrideWise<paddle::platform::CPUDeviceContext, T, Func> {
    void operator()(DDim in_dims, DDim out_dims,
                    std::vector<int>& ksize, std::vector<int>& strides, std::vector<int>& paddings,
                    const T* src, T* target, int src_stride, int target_stride, Func visitor) {
        const int share_size = in_dims[0];
        const int batch_size = in_dims[1];
        const int channel_size = in_dims[2];
        const int input_height = in_dims[3];
        const int input_width = in_dims[4];
        const int out_height = out_dims[3];
        const int out_width = out_dims[4];
        const int out_mat_numel = out_height * out_width;

        const int ksize_height = ksize[0];
        const int ksize_width = ksize[1];
        const int filter_numel = ksize_height * ksize_width;
        const int stride_height = strides[0];
        const int stride_width = strides[1];
        const int padding_height = paddings[0];
        const int padding_width = paddings[1];

        int hstart, hend;
        int wstart, wend;

        int idx = 0;
        while (idx++ < batch_size * channel_size) {
            for (size_t ph = 0; ph < out_height; ++ph) {
                hstart =  ph * stride_height - padding_height;
                hend = std::min(hstart + ksize_height, input_height);
                hstart = std::max(hstart, 0);

                for (size_t pw = 0; pw < out_width; ++pw) {
                    wstart = pw * stride_width - padding_width;
                    wend = std::min(wstart + ksize_width, input_width);
                    wstart = std::max(wstart, 0);

                    visitor(ph, pw, input_height, input_width, out_height, out_width, hstart, hend,
                               wstart, wend, src, target);
                }
            }
            src += src_stride;
            target += target_stride;
        }
}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    mpc_pool2d, ops::MpcPoolOp, ops::MpcPool2dOpMaker, ops::MpcPoolOpInferVarType,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(mpc_pool2d_grad, ops::MpcPoolOpGrad);

#ifdef USE_CUDA

#else // USE_CUDA

REGISTER_OP_CPU_KERNEL(
    mpc_pool2d, ops::MpcPoolKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    mpc_pool2d_grad, ops::MpcPoolGradKernel<paddle::platform::CPUDeviceContext, int64_t>);

#endif // USE_CUDA
