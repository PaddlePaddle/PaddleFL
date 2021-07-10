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

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

#include "./math/im2col.h"
#include "./math/vol2col.h"
#include "./math/math_function.h"
#include "mpc_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int kConvMKLDNNFP32 = 1;
constexpr int kConvMKLDNNINT8 = 2;
constexpr int MaxKeyLength = 256;

// Base convolution operator definations for other conv
// like operators to reuse the implementation.
inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size, 0,
      platform::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But recieved: output's size is %d. The output's size is computed by "
          "((input_size + 2 * padding - (dilation * (filter_size - 1) + 1)) / "
          "stride + 1), where input_size is %d, padding is %d, "
          "filter_size is %d, dilation is %d, stride is %d.",
          output_size, input_size, padding, filter_size, dilation, stride));

  return output_size;
}

inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding_1, int padding_2, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + padding_1 + padding_2 - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size, 0,
      platform::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But recieved: output's size is %d. The output's size is computed by "
          "((input_size + padding_1 + padding_2 - (dilation * (filter_size - "
          "1) + 1)) / stride + 1), where input_size is %d, padding is "
          "(%d, %d), filter_size is %d, dilation is %d, stride is %d.",
          output_size, input_size, padding_1, padding_2, filter_size, dilation,
          stride));

  return output_size;
}

template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string& padding_algorithm,
                                     const framework::DDim data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = framework::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2, paddings->size(),
        platform::errors::InvalidArgument(
            "Attribute padding's size should be the same or twice as the "
            "input's dimension. "
            "But recieved: padding's size is %d, padding is [%s]; input's "
            "dimension is %d, input's shape is [%s].",
            paddings->size(), framework::make_ddim(*paddings), data_dims.size(),
            data_dims));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = 1;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    // extra 1 for share dim
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2 + 1]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  if (paddings.size() != strides.size()) {
    for (size_t j = 0; j < paddings.size(); ++j) {
      padding_0 = padding_0 && (paddings[j] == 0);
    }
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

template <typename DeviceContext, typename T>
inline void ResizeToChannelFirst(const framework::ExecutionContext& context,
                                 const Tensor* input,
                                 Tensor* transformed_input,
                                 bool is_output = false) {
  // extra 1 for leading share dim S
  int dim = input->dims().size() - 2 - 1;
  if (dim == 3) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = framework::vectorize(input->dims());
    if (is_output) {
        // same as paddle, resize output of conv op
        // SNDHWC -> SNCDHW
        // all for simulate paddle conv op (plaintext)'s behavior
        in_dims_vec[0] = input->dims()[0];
        in_dims_vec[1] = input->dims()[1];
        in_dims_vec[2] = input->dims()[5];
        in_dims_vec[3] = input->dims()[2];
        in_dims_vec[4] = input->dims()[3];
        in_dims_vec[5] = input->dims()[4];
    } else {
        // SNDHWC -> NCSDHW
        in_dims_vec[0] = input->dims()[1];
        in_dims_vec[1] = input->dims()[5];
        in_dims_vec[2] = input->dims()[0];
        in_dims_vec[3] = input->dims()[2];
        in_dims_vec[4] = input->dims()[3];
        in_dims_vec[5] = input->dims()[4];
    }
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());

  } else if (dim == 2) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = framework::vectorize(input->dims());
    if (is_output) {
        in_dims_vec[0] = input->dims()[0];
        in_dims_vec[1] = input->dims()[1];
        in_dims_vec[2] = input->dims()[4];
        in_dims_vec[3] = input->dims()[2];
        in_dims_vec[4] = input->dims()[3];
    } else {
        // SNHWC -> NCSHW
        in_dims_vec[0] = input->dims()[1];
        in_dims_vec[1] = input->dims()[4];
        in_dims_vec[2] = input->dims()[0];
        in_dims_vec[3] = input->dims()[2];
        in_dims_vec[4] = input->dims()[3];
    }
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());
  }
}

template <typename DeviceContext, typename T>
inline void ResizeToChannelLast(const framework::ExecutionContext& context,
                                const Tensor* input,
                                Tensor* transformed_input) {
  // extra 1 for leading share dim S
  int dim = input->dims().size() - 2 - 1;
  if (dim == 3) {
    // input
    transformed_input->Resize(input->dims());

    // NCSDHW -> SNDHWC
    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[0] = input->dims()[2];
    in_dims_vec[1] = input->dims()[0];
    in_dims_vec[2] = input->dims()[3];
    in_dims_vec[3] = input->dims()[4];
    in_dims_vec[4] = input->dims()[5];
    in_dims_vec[5] = input->dims()[1];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());

  } else if (dim == 2) {
    // input
    transformed_input->Resize(input->dims());

    // NCSHW -> SNHWC
    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[0] = input->dims()[2];
    in_dims_vec[1] = input->dims()[0];
    in_dims_vec[2] = input->dims()[3];
    in_dims_vec[3] = input->dims()[4];
    in_dims_vec[4] = input->dims()[1];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());
  }
}

template <typename DeviceContext, typename T>
inline void ResizeToShareLast(const framework::ExecutionContext& context,
                                        const Tensor* input,
                                        Tensor* transformed_input) {
    transformed_input->Resize(input->dims());

    // SNC.. -> NCS..
    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[0] = input->dims()[1];
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[0];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());
}

template <typename DeviceContext, typename T>
inline void ResizeToShareFirst(const framework::ExecutionContext& context,
                                        const Tensor* input,
                                        Tensor* transformed_input) {
    transformed_input->Resize(input->dims());

    // NCS.. -> SNC..
    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[0] = input->dims()[2];
    in_dims_vec[1] = input->dims()[0];
    in_dims_vec[2] = input->dims()[1];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());
}

template <typename DeviceContext, typename T>
inline void TransToChannelFirst(const framework::ExecutionContext& context,
                                const Tensor* input,
                                Tensor* transformed_input,
                                bool is_output = false) {
  // extra 1 for leading share dim
  // swap share and batch_size
  int dim = input->dims().size() - 2 - 1;
  if (dim == 3) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis;
    if (is_output) {
        axis = std::vector<int>{0, 1, 5, 2, 3, 4};
    } else {
        axis = std::vector<int>{1, 5, 0, 2, 3, 4};
    }
    math::Transpose<DeviceContext, T, 6> trans6;
    trans6(dev_ctx, *input, transformed_input, axis);

  } else if (dim == 2) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{1, 4, 0, 2, 3};
    if (is_output) {
        axis = std::vector<int>{0, 1, 4, 2, 3};
    } else {
        axis = std::vector<int>{1, 4, 0, 2, 3};
    }
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(dev_ctx, *input, transformed_input, axis);
  }
}

template <typename DeviceContext, typename T>
inline void TransToChannelLast(const framework::ExecutionContext& context,
                               const Tensor* input, Tensor* transformed_input) {
  // extra 1 for leading share dim
  // swap share and batch_size
  int dim = input->dims().size() - 2 - 1;
  if (dim == 3) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 1, 3, 4, 5, 2};
    math::Transpose<DeviceContext, T, 6> trans6;
    trans6(dev_ctx, *input, transformed_input, axis);

  } else if (dim == 2) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 1, 3, 4, 2};
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(dev_ctx, *input, transformed_input, axis);
  }
}

template <typename DeviceContext, typename T>
inline void TransToShareFirst(const framework::ExecutionContext& context,
                              const Tensor* input, Tensor* transformed_input) {
  int dim = input->dims().size();

  PADDLE_ENFORCE_GT(
      dim, 3,
      platform::errors::InvalidArgument(
          "The input's dim is expected to be greater than 4."));

  std::vector<int> axis(dim);
  for (size_t i = 3; i < dim; ++i) {
      axis[i] = i;
  }
  // share
  axis[0] = 2;
  // N
  axis[1] = 0;
  // C
  axis[2] = 1;

  auto& dev_ctx = context.template device_context<DeviceContext>();

  switch(dim) {

  case 4:
    math::Transpose<DeviceContext, T, 4> trans4;
    trans4(dev_ctx, *input, transformed_input, axis);
    break;

  case 5:
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(dev_ctx, *input, transformed_input, axis);
    break;

  case 6:
    math::Transpose<DeviceContext, T, 6> trans6;
    trans6(dev_ctx, *input, transformed_input, axis);
    break;

  default:
    PADDLE_ENFORCE_LT(
        dim, 7, platform::errors::InvalidArgument(
            "The input's dim greater than 6 not supported yet. "));
  }
}

template <typename DeviceContext, typename T>
inline void TransToShareLast(const framework::ExecutionContext& context,
                              const Tensor* input, Tensor* transformed_input) {
  int dim = input->dims().size();

  PADDLE_ENFORCE_GT(
      dim, 4,
      platform::errors::InvalidArgument(
          "The input's dim is expected to be greater than 4."));

  std::vector<int> axis(dim);
  for (size_t i = 3; i < dim; ++i) {
      axis[i] = i;
  }
  // SNC -> NCS
  axis[0] = 1;
  axis[1] = 2;
  axis[2] = 0;

  auto& dev_ctx = context.template device_context<DeviceContext>();

  switch(dim) {

  case 5:
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(dev_ctx, *input, transformed_input, axis);
    break;

  case 6:
    math::Transpose<DeviceContext, T, 6> trans6;
    trans6(dev_ctx, *input, transformed_input, axis);
    break;

  default:
    PADDLE_ENFORCE_LT(
        dim, 7, platform::errors::InvalidArgument(
            "The input's dim greater than 6 not supported yet. "));
  }
}
template <typename DeviceContext, typename T>
inline void TransToBatchFirst(const framework::ExecutionContext& context,
                              const Tensor* input, Tensor* transformed_input) {
  int dim = input->dims().size();

  PADDLE_ENFORCE_GT(
      dim, 4,
      platform::errors::InvalidArgument(
          "The input's dim is expected to be greater than 4."));

  std::vector<int> axis(dim);
  for (size_t i = 3; i < dim; ++i) {
      axis[i] = i;
  }
  // N
  axis[0] = 1;
  // C
  axis[1] = 2;
  // share
  axis[2] = 0;

  auto& dev_ctx = context.template device_context<DeviceContext>();

  switch(dim) {

  case 5:
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(dev_ctx, *input, transformed_input, axis);
    break;

  case 6:
    math::Transpose<DeviceContext, T, 6> trans6;
    trans6(dev_ctx, *input, transformed_input, axis);
    break;

  default:
    PADDLE_ENFORCE_LT(
        dim, 7, platform::errors::InvalidArgument(
            "The input's dim greater than 6 not supported yet. "));
  }
}

template <typename DeviceContext, typename T>
inline void ResizeToSwapedLeadingDims(const framework::ExecutionContext& context,
                                      const Tensor* input,
                                Tensor* transformed_input) {
    transformed_input->Resize(input->dims());

    // NS.. -> SN..
    // or CS.. -> SC..
    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[0] = input->dims()[1];
    in_dims_vec[1] = input->dims()[0];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());
}

template <typename DeviceContext, typename T>
void TransToSwapedLeadingDims(const framework::ExecutionContext& context,
                       const Tensor* input,
                       Tensor* output){
    output->Resize(input->dims());
    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[0] = input->dims()[1];
    in_dims_vec[1] = input->dims()[0];
    output->Resize(framework::make_ddim(in_dims_vec));
    output->mutable_data<T>(context.GetPlace());

    const int dim = input->dims().size();

    std::vector<int> axis(dim);
    for (size_t i = 0; i < dim; ++i) {
        axis[i] = i;
    }
    axis[0] = 1;
    axis[1] = 0;

    auto& dev_ctx = context.template device_context<DeviceContext>();

    switch(dim) {

    case 3:
      math::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, *input, output, axis);
      break;

    case 4:
      math::Transpose<DeviceContext, T, 4> trans4;
      trans4(dev_ctx, *input, output, axis);
      break;

    case 5:
      math::Transpose<DeviceContext, T, 5> trans5;
      trans5(dev_ctx, *input, output, axis);
      break;

    case 6:
      math::Transpose<DeviceContext, T, 6> trans6;
      trans6(dev_ctx, *input, output, axis);
      break;

    default:
      PADDLE_ENFORCE_GT(
          dim, 2, platform::errors::InvalidArgument(
              "The input's dim less than 3 not supported yet. "));
      PADDLE_ENFORCE_LT(
          dim, 7, platform::errors::InvalidArgument(
              "The input's dim greater than 6 not supported yet. "));
    }
    return;
}

template <typename DeviceContext, typename T, typename Func>
void SharesToCols(const framework::ExecutionContext& context,
      const Tensor* input,
      const std::vector<int>& dilations,
      const std::vector<int>& strides,
      const std::vector<int>& paddings,
      Tensor* col, Func data2col) {
    // // input: CSHW or CSDHW, S for share dim

    framework::DDim in_plain_dim =
        framework::slice_ddim(input->dims(), 1, input->dims().size());
    framework::DDim col_plain_dim =
        framework::slice_ddim(col->dims(), 1, col->dims().size());

    auto& dev_ctx = context.template device_context<DeviceContext>();

    const int share_size = input->dims()[0];
    for (size_t i = 0; i < share_size; ++i) {
        Tensor share = input->Slice(i, i + 1).Resize(in_plain_dim);
        Tensor col_share = col->Slice(i, i + 1).Resize(col_plain_dim);
        data2col(dev_ctx, share, dilations, strides, paddings, &col_share);
    }
}

template <typename DeviceContext, typename T>
Tensor SwapedLeadingDims(const framework::ExecutionContext& context,
                         const Tensor* input) {
    Tensor output(input->type());

    ResizeToSwapedLeadingDims<DeviceContext, T>(context, input,
                                                &output);
    TransToSwapedLeadingDims<DeviceContext, T>(context, input,
                                               &output);
    return output;
}

template <typename DeviceContext, typename T>
Tensor TransposeMpcMat(const framework::ExecutionContext& context,
                       const Tensor* input) {
    Tensor output(input->type());

    auto in_dims_vec = framework::vectorize(input->dims());

    PADDLE_ENFORCE_EQ(
        in_dims_vec.size(), 3, platform::errors::InvalidArgument(
            "The input's dim should be 3. "));
    in_dims_vec[0] = input->dims()[0];
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[1];
    output.Resize(framework::make_ddim(in_dims_vec));
    output.mutable_data<T>(context.GetPlace());

    std::vector<int> axis(3);
    axis[0] = 0;
    axis[1] = 2;
    axis[2] = 1;

    auto& dev_ctx = context.template device_context<DeviceContext>();

    math::Transpose<DeviceContext, T, 3> trans3;
    trans3(dev_ctx, *input, &output, axis);

    return output;
}

// Define Op classes in .h file so that other conv
// operator implementations can reuse the code.
class Conv2DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

class ConvOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"Input", /*->*/ "Output"}};
    return m;
  }
};

template <typename DeviceContext, typename T>
struct CopyData {
    void operator()(T* dst, const T* src, size_t numel);
};

class ConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    std::vector<int64_t> output_shape = ComputeOutputShape(ctx);

    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output", "Conv");
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
    ctx->ShareLoD("Input", "Output");
  }

 protected:
  std::vector<int64_t> ComputeOutputShape(
      framework::InferShapeContext* ctx) const;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class ConvOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

// TODO: add conv double grad

template <typename DeviceContext, typename T>
class GemmConvKernel : public MpcOpKernel<T> {
 public:
  void ComputeImpl(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    const int groups = context.Attr<int>("groups");
    const std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    Tensor transformed_input(input->type());
    Tensor transformed_output(output->type());

    if (channel_last) {
      ResizeToChannelFirst<DeviceContext, T>(context, input,
                                             &transformed_input);
      TransToChannelFirst<DeviceContext, T>(context, input, &transformed_input);

      ResizeToChannelFirst<DeviceContext, T>(context, output,
                                             &transformed_output, true);

    } else {
      ResizeToShareLast<DeviceContext, T>(context, input,
                                          &transformed_input);
      TransToShareLast<DeviceContext, T>(context, input, &transformed_input);

      transformed_output = *output;
    }

    // update padding and dilation
    auto trans_in_dims = transformed_input.dims();
    auto filter_dims = filter.dims();

    // extra 1 for share dim
    framework::DDim in_data_dims =
        framework::slice_ddim(trans_in_dims, 2 + 1, trans_in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2 + 1, filter_dims.size());

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    auto& dev_ctx = context.template device_context<DeviceContext>();

    const int batch_size = static_cast<int>(transformed_input.dims()[0]);

    // filter_shape_vec:
    // {k_share, k_o, k_i, k_h, k_w} or {k_share, k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));

    // output_shape_vec:
    // {o_n, o_c, o_share, o_h, o_w} or {o_n, o_c, o_share, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(transformed_output.dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec:
    // {i_s, i_c/g, k_h, k_w, o_h, o_w} or {i_s, i_c/g, k_d, k_h, k_w,
    // o_d, o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2 - 1;

    std::vector<int64_t> col_shape_vec(2 + 2 * data_dim);
    col_shape_vec[0] = trans_in_dims[2];
    col_shape_vec[1] = trans_in_dims[1] / groups;

    std::vector<int64_t> col_matrix_shape_vec(3);
    col_matrix_shape_vec[0] = col_shape_vec[0];
    col_matrix_shape_vec[1] = col_shape_vec[1];
    col_matrix_shape_vec[2] = 1;
    // use col_matrix_shape in the gemm calculation
    // size:
    // (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w, o_d * o_h *
    // o_w)

    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 2] = filter_shape_vec[j + 3];
      col_shape_vec[j + 2 + data_dim] = output_shape_vec[j + 3];
      col_matrix_shape_vec[1] *= filter_shape_vec[j + 3];
      col_matrix_shape_vec[2] *= output_shape_vec[j + 3];
    }

    framework::DDim col_shape(framework::make_ddim(col_shape_vec));

    framework::DDim col_matrix_shape(framework::make_ddim(col_matrix_shape_vec));

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);

    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      col = context.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    // with share dim
    framework::DDim in_matrix_shape = framework::slice_ddim(
        transformed_input.dims(), 1, transformed_input.dims().size());

    // SOIHW or SOIDHW
    framework::DDim filter_matrix_shape = {filter.dims()[0], filter.dims()[1],
                                           filter.numel() / (filter.dims()[0] * filter.dims()[1]) };
    filter.Resize(filter_matrix_shape);

    int in_step = static_cast<int>(transformed_input.dims()[1]) / groups;
    int out_step = static_cast<int>(transformed_output.dims()[2]) / groups;

    // S, N*groups, C/groups, H*W or D*H*W
    framework::DDim output_matrix_shape = {
        transformed_output.dims()[0],
        batch_size * groups,
        out_step,
        transformed_output.numel() /
            (transformed_output.dims()[0]
             * transformed_output.dims()[1]
             * transformed_output.dims()[2])};

    // convolution operator: im2col(or vol2col) + gemm
    math::Vol2ColFunctor<DeviceContext, T> vol2col;
    math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;

    Tensor batched_col;
    Tensor batched_filter;

    batched_col.mutable_data<T>(framework::make_ddim({batch_size * groups,
                                                     col_matrix_shape[0],
                                                     col_matrix_shape[1],
                                                     col_matrix_shape[2]}),
                                context.GetPlace(), 0);

    batched_filter.mutable_data<T>(framework::make_ddim({batch_size,
                                                        filter_matrix_shape[0],
                                                        groups,
                                                        out_step,
                                                        filter_matrix_shape[2],
                                                        }),
                                   context.GetPlace(), 0);

    auto original_out_dims = transformed_output.dims();

    transformed_output.Resize(output_matrix_shape);
    transformed_output.mutable_data<T>(context.GetPlace());

    auto copy_functor = CopyData<DeviceContext, T>();

    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch =
          transformed_input.Slice(i, i + 1).Resize(in_matrix_shape);

      Tensor filter_slice = batched_filter.Slice(i, i + 1);

      copy_functor(filter_slice.data<T>(),
                  filter.data<T>(), filter.numel());

      for (int g = 0; g < groups; g++) {
        Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

        Tensor in_slice_ = SwapedLeadingDims<DeviceContext, T>(context, &in_slice);

        if (!is_expand) {
          col.ShareDataWith(in_slice_);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          SharesToCols<DeviceContext, T>(context, &in_slice_, dilations, strides,
                 std::vector<int>{paddings[0], paddings[2], paddings[1],
                 paddings[3]}, &col, im2col);
        } else if (data_dim == 3U) {
          SharesToCols<DeviceContext, T>(context, &in_slice_, dilations, strides, paddings, &col, vol2col);
        }

        size_t col_matrix_size = col_matrix.numel();
        size_t col_group_size = col_matrix_size * groups;

        copy_functor(batched_col.template data<T>()
                    + i * col_group_size + g * col_matrix_size,
                    col_matrix.template data<T>(),
                    col_matrix_size);

      }
    }
    Tensor batched_col_ = SwapedLeadingDims<DeviceContext, T>(context, &batched_col);
    Tensor batched_fil_ = SwapedLeadingDims<DeviceContext, T>(context, &batched_filter);
    batched_fil_.Resize(framework::make_ddim({filter_matrix_shape[0],
                                                batch_size * groups,
                                                out_step,
                                                filter_matrix_shape[2],
                                                }));
    // TransToShareFirst<DeviceContext, T>(context, &batched_filter, &batched_fil_);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->matmul(
                    &batched_fil_, &batched_col_, &transformed_output);

    transformed_output.Resize(original_out_dims);

    if (channel_last) {
      TransToChannelLast<DeviceContext, T>(context, &transformed_output,
                                           output);
    }
  }
};

template <typename DeviceContext, typename T>
class GemmConvGradKernel : public MpcOpKernel<T> {
 public:
  void ComputeImpl(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    // The filter and filter_grad will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");

    if (!input_grad && !filter_grad) return;

    int groups = context.Attr<int>("groups");
    const std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    Tensor transformed_input(input->type());
    Tensor transformed_output_grad(output_grad->type());

    if (channel_last) {
      ResizeToChannelFirst<DeviceContext, T>(context, input,
                                             &transformed_input);
      TransToChannelFirst<DeviceContext, T>(context, input, &transformed_input);

      ResizeToChannelFirst<DeviceContext, T>(context, output_grad,
                                             &transformed_output_grad, true);
      TransToChannelFirst<DeviceContext, T>(context, output_grad,
                                            &transformed_output_grad, true);
    } else {
      ResizeToShareLast<DeviceContext, T>(context, input,
                                             &transformed_input);
      TransToShareLast<DeviceContext, T>(context, input, &transformed_input);

      transformed_output_grad = *output_grad;
    }

    // update padding and dilation
    auto in_dims = transformed_input.dims();
    auto filter_dims = filter.dims();
    // extra 1 for share dim
    framework::DDim in_data_dims =
        framework::slice_ddim(in_dims, 2 + 1, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2 + 1, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    const int batch_size = static_cast<int>(transformed_input.dims()[0]);

    auto& dev_ctx = context.template device_context<DeviceContext>();

    // filter_shape_vec: {k_share, k_o, k_i, k_h, k_w} or {k_share, k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    // output_shape_vec: {o_n, o_c, o_share, o_h, o_w} or {o_n, o_c, o_share, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(transformed_output_grad.dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
    // o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2 - 1;
    std::vector<int64_t> col_shape_vec(2 + 2 * data_dim);
    col_shape_vec[0] = in_dims[2];
    col_shape_vec[1] = in_dims[1] / groups;

    std::vector<int64_t> col_matrix_shape_vec(3);
    col_matrix_shape_vec[0] = col_shape_vec[0];
    col_matrix_shape_vec[1] = col_shape_vec[1];
    col_matrix_shape_vec[2] = 1;
    // use col_matrix_shape in the gemm calculation
    // size:
    // (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w, o_d * o_h *
    // o_w)
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 2] = filter_shape_vec[j + 3];
      col_shape_vec[j + 2 + data_dim] = output_shape_vec[j + 3];
      col_matrix_shape_vec[1] *= filter_shape_vec[j + 3];
      col_matrix_shape_vec[2] *= output_shape_vec[j + 3];
    }

    framework::DDim col_shape(framework::make_ddim(col_shape_vec));
    framework::DDim col_matrix_shape(framework::make_ddim(col_matrix_shape_vec));

    // with share dim
    framework::DDim input_shape = framework::slice_ddim(
        transformed_input.dims(), 1, transformed_input.dims().size());

    // SOIHW or SOIDHW
    framework::DDim filter_matrix_shape = {filter.dims()[0], filter.dims()[1],
                                           filter.numel() / (filter.dims()[0] * filter.dims()[1]) };

    filter.Resize(filter_matrix_shape);

    // convolution backward input operator:  gemm + col2im(or col2vol)
    // convolution backward weight operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(transformed_input.dims()[1]) / groups;
    int out_step = static_cast<int>(transformed_output_grad.dims()[2]) / groups;

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);

    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      col = context.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    math::SetConstant<DeviceContext, T> set_zero;

    Tensor batched_filter;

    batched_filter.mutable_data<T>(framework::make_ddim({batch_size,
                                                        filter_matrix_shape[0],
                                                        groups,
                                                        out_step,
                                                        filter_matrix_shape[2],
                                                        }),
                                   context.GetPlace(), 0);

    auto copy_functor = CopyData<DeviceContext, T>();
// #pragma omp for
    for (int i = 0; i < batch_size; i++) {

      Tensor filter_slice = batched_filter.Slice(i, i + 1);

      copy_functor(filter_slice.data<T>(),
                  filter.data<T>(), filter.numel());
    }

    Tensor batched_fil_ = SwapedLeadingDims<DeviceContext, T>(context, &batched_filter);

    batched_fil_.Resize(framework::make_ddim({filter_matrix_shape[0],
                                                batch_size * groups,
                                                out_step,
                                                filter_matrix_shape[2],
                                                }));

    transformed_output_grad.Resize(framework::make_ddim({
        transformed_output_grad.dims()[0],
        batch_size * groups,
        out_step,
        transformed_output_grad.numel() / (
            transformed_output_grad.dims()[0] *
            transformed_output_grad.dims()[1] *
            transformed_output_grad.dims()[2])
        }));

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      Tensor transformed_input_grad(input_grad->type());
      if (channel_last) {
        ResizeToChannelFirst<DeviceContext, T>(context, input_grad,
                                               &transformed_input_grad);

      } else {
        ResizeToShareLast<DeviceContext, T>(context, input_grad,
                                            &transformed_input_grad);
      }

      // if is_expand is false, the operation of set_zero is unnecessary,
      // because math::matmul will reset input_grad.
      if (is_expand) {
        set_zero(dev_ctx, &transformed_input_grad, static_cast<T>(0));
      }

      Tensor batched_gemm(input_grad->type());
      batched_gemm.Resize(framework::make_ddim({col_matrix_shape[0],
                                               batch_size * groups,
                                               col_matrix_shape[1],
                                               col_matrix_shape[2],
                                               }));
      batched_gemm.mutable_data<T>(context.GetPlace());

      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->matmul(
          &batched_fil_, &transformed_output_grad, &batched_gemm, true, false);

      batched_gemm = SwapedLeadingDims<DeviceContext, T>(
          context, &batched_gemm);

      batched_gemm.Resize(framework::make_ddim({batch_size,
                                                groups,
                                                col_matrix_shape[0],
                                                col_matrix_shape[1],
                                                col_matrix_shape[2],
                                                }));


      math::Col2VolFunctor<DeviceContext, T> col2vol;
      math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;

      for (int i = 0; i < batch_size; i++) {
        Tensor in_grad_batch =
            transformed_input_grad.Slice(i, i + 1).Resize(input_shape);
        Tensor gemm_group =
            batched_gemm.Slice(i, i + 1).Resize(framework::make_ddim(
                    {groups, col_matrix_shape[0],
                    col_matrix_shape[1], col_matrix_shape[2], }));
        for (int g = 0; g < groups; g++) {
          Tensor in_grad_slice =
              in_grad_batch.Slice(g * in_step, (g + 1) * in_step);

          Tensor gemm =
              gemm_group.Slice(g, g + 1).Resize(framework::make_ddim(
                      { col_matrix_shape[0], col_matrix_shape[1], col_matrix_shape[2], }));

          Tensor im_ =
              SwapedLeadingDims<DeviceContext, T>(context, &in_grad_slice);

          if (!is_expand) {
            gemm.Resize(im_.dims());
          } else {
            gemm.Resize(col.dims());
          }

          if (is_expand && data_dim == 2U) {
            SharesToCols<DeviceContext, T>(context, &gemm, dilations, strides,
                   std::vector<int>{paddings[0], paddings[2], paddings[1],
                                    paddings[3]},
                   &im_, col2im);
          } else if (is_expand && data_dim == 3U) {
              SharesToCols<DeviceContext, T>(context, &gemm, dilations, strides, paddings, &im_, col2vol);
          }
          TransToSwapedLeadingDims<DeviceContext, T>(context, is_expand ? &im_ : &gemm, &in_grad_slice);
        }
      }
      if (channel_last) {
        TransToChannelLast<DeviceContext, T>(context, &transformed_input_grad,
                                             input_grad);
      } else {
        TransToShareFirst<DeviceContext, T>(context, &transformed_input_grad,
                                            input_grad);
      }
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      auto filter_grad_dims = filter_grad->dims();

      Tensor filter_grad_;

      filter_grad_.mutable_data<T>(framework::make_ddim({filter_matrix_shape[0],
                                               batch_size * groups,
                                               out_step,
                                               filter_matrix_shape[2]
                                               }), context.GetPlace(), 0);

      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;

      Tensor batched_col;

      batched_col.mutable_data<T>(framework::make_ddim({batch_size * groups,
                                                       col_matrix_shape[0],
                                                       col_matrix_shape[1],
                                                       col_matrix_shape[2]}),
                                  context.GetPlace(), 0);

      for (int i = 0; i < batch_size; i++) {
        Tensor in_batch = transformed_input.Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

          Tensor in_slice_ = SwapedLeadingDims<DeviceContext, T>(context, &in_slice);
          if (!is_expand) {
            col.ShareDataWith(in_slice_);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            SharesToCols<DeviceContext, T>(context, &in_slice_, dilations, strides,
                                           std::vector<int>{paddings[0], paddings[2], paddings[1],
                                           paddings[3]}, &col, im2col);

          } else if (data_dim == 3U) {
            SharesToCols<DeviceContext, T>(context, &in_slice_, dilations, strides, paddings, &col, vol2col);
          }
          size_t col_matrix_size = col_matrix.numel();
          size_t col_group_size = col_matrix_size * groups;

          copy_functor(batched_col.template data<T>()
                      + i * col_group_size + g * col_matrix_size,
                      col_matrix.template data<T>(),
                      col_matrix_size);
        }
      }
      Tensor batched_col_ = SwapedLeadingDims<DeviceContext, T>(context, &batched_col);
      // gemm

      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->matmul(
          &transformed_output_grad, &batched_col_, &filter_grad_, 0, 1);

      filter_grad_.Resize(framework::make_ddim({filter_matrix_shape[0],
                                               batch_size,
                                               groups,
                                               out_step,
                                               filter_matrix_shape[2]
                                               }));

      using EigenTensor5 = paddle::framework::EigenTensor<T, 5>;
      using EigenTensor4 = paddle::framework::EigenTensor<T, 4>;

      filter_grad->Resize(framework::make_ddim({filter_matrix_shape[0],
                                               groups,
                                               out_step,
                                               filter_matrix_shape[2]
                                               }));

      auto eigen_filter_grad_ = EigenTensor5::From(filter_grad_);
      auto eigen_filter_grad = EigenTensor4::From(*filter_grad);

      eigen_filter_grad.device(*dev_ctx.eigen_device()) =
          eigen_filter_grad_.sum(Eigen::array<int,1>({1}));

      filter_grad->Resize(filter_grad_dims);

    }
  }
};

}  // namespace operators
}  // namespace paddle
