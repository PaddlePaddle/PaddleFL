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

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "mpc_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings, const bool global_pooling,
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
        PADDLE_ENFORCE_EQ(data_dims.size() * 2, paddings->size(),
                          "Paddings size should be the same or twice as the pooling size.");
    }

    // when padding_algorithm is "VALID" or "SAME"
    if (padding_algorithm == "SAME") {
        for (int i = 0; i < data_dims.size(); ++i) {
            T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
            T pad_sum = std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i], static_cast<T>(0));
            T pad_0 = pad_sum / 2;
            T pad_1 = pad_sum - pad_0;
            *(paddings->begin() + i * 2) = pad_0;
            *(paddings->begin() + i * 2 + 1) = pad_1;
        }
    } else if (padding_algorithm == "VALID") {
        for (auto it = paddings->begin(); it != paddings->end(); it++) {
            *it = 0;
        }
    }

    // if global_pooling == true, padding will be ignore
    if (global_pooling) {
        for (auto it = paddings->begin(); it != paddings->end(); it++) {
            *it = 0;
        }
    }
}

template <typename T = int>
inline void UpdateKsize(std::vector<T>* ksize,
                        const framework::DDim data_dims) {
    ksize->resize(static_cast<size_t>(data_dims.size()));
    for (size_t i = 0; i < ksize->size(); ++i) {
        *(ksize->begin() + i) = static_cast<T>(data_dims[i]);
    }
}

template <typename DeviceContext, typename T, typename Func>
struct VisitDataStrideWise {
    void operator()(DDim in_dims, DDim out_dims,
                    std::vector<int>& ksize, std::vector<int>& strides, std::vector<int>& paddings,
                    const T* src, T* target, int src_stride, int target_stride, Func vistor);
};

template <typename DeviceContext, typename T>
class MpcPoolKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &context) const override {

        const Tensor* in_x = context.Input<Tensor>("X");
        Tensor* out = context.Output<Tensor>("Out");
        Tensor* out_one_hot_tensor = context.Output<Tensor>("One_hot_tensor");

        std::string pooling_type = context.Attr<std::string>("pooling_type");
        std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
        std::vector<int> strides = context.Attr<std::vector<int>>("strides");
        std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
        std::string data_format = context.Attr<std::string>("data_format"); // NCHW
        bool global_pooling = context.Attr<bool>("global_pooling");
        std::string padding_algorithm =
            context.Attr<std::string>("padding_algorithm");

        const T* in_x_data = in_x->data<T>();
        T* output_data = out->mutable_data<T>(context.GetPlace());
        T* one_hot_tensor_data = out_one_hot_tensor->mutable_data<T>(context.GetPlace());

        // update paddings
        auto in_x_dims = in_x->dims();
        auto out_dims = out->dims();

        const int input_stride = in_x_dims[3] * in_x_dims[4];
        const int output_stride = out_dims[3] * out_dims[4];
        const int one_hot_tensor_stride = ksize[0] * ksize[1] * out_dims[3] * out_dims[4];

        // create temp tensor
        auto& dev_ctx = context.template device_context<DeviceContext>();
        Tensor input2col = context.AllocateTmpTensor<T, DeviceContext>(out_one_hot_tensor->dims(), dev_ctx);
        T* input2col_data = input2col.data<T>();
        math::SetConstant<DeviceContext, T> set_constant;
        set_constant(dev_ctx, &input2col, 0);

        framework::DDim data_dims;
        data_dims = framework::slice_ddim(in_x_dims, 3, in_x_dims.size());

        // update padding => h, w
        UpdatePadding(&paddings, global_pooling, padding_algorithm,
                      data_dims, strides, ksize);
        if (data_dims.size() * 2 == static_cast<int>(paddings.size())) {
            for (int i = 0; i < data_dims.size(); ++i) {
               paddings.erase(paddings.begin() + i + 1);
            }
        }

        if (global_pooling) {
            UpdateKsize(&ksize, data_dims);
        }

        // share0, share1
        const int input_plaintext_size = in_x->numel() / 2;
        const int input2col_plaintext_size = out_one_hot_tensor->numel() / 2;

        // im2col
        auto get_im2col = [input2col_plaintext_size, input_plaintext_size]
#ifdef __NVCC__
            __host__ __device__
#endif
            (int ph, int pw, int input_height, int input_width, int out_height, int out_width,
             int hstart, int hend, int wstart, int wend, const T* src, T* target) {

            size_t out_index = ph * out_width + pw;
            size_t offset = out_height * out_width;
            size_t index = 0;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    target[out_index + index * offset] = src[h * input_width + w]; // share0
                    target[out_index + index * offset + input2col_plaintext_size] =
                    src[h * input_width + w + input_plaintext_size]; // share1
                    ++index;
                }
            }
        };

        auto visit_functor = VisitDataStrideWise<DeviceContext, T, decltype(get_im2col)>();

        // input2col
        // convert in_x_data (S, B, C, H, W) into (S, B, C, filter_size * filter_size, H_output * W_output)
        visit_functor(in_x_dims, out_dims, ksize, strides, paddings, in_x_data, input2col_data, input_stride, one_hot_tensor_stride, get_im2col);

        const T* input2col_data2 = input2col.data<T>();

        // maxpooling(input2col_trans), return(max2col, out_one_hot_tensor_trans)
        // input2col_trans: (S, filter_size * filter_size, B, C, H_output * W_output)
        // max2col: (S, , B, C, H_output * W_output)
        // out_one_hot_tensor_trans: (S, filter_size * filter_size, B, C, H_output * W_output)
        Tensor input2col_trans;
        DDim in2col_dims = input2col.dims();
        T* input2col_trans_data = input2col_trans.mutable_data<T>(in2col_dims, context.GetPlace());
        input2col_trans.Resize({in2col_dims[0], in2col_dims[3], in2col_dims[1], in2col_dims[2], in2col_dims[4]});

        Tensor max2col;
        max2col.ShareDataWith(*out);
        max2col.Resize({in2col_dims[0], 1, in2col_dims[1], in2col_dims[2], in2col_dims[4]});

        Tensor out_one_hot_tensor_trans;
        out_one_hot_tensor_trans.mutable_data<T>(out_one_hot_tensor->dims(), context.GetPlace());
        out_one_hot_tensor_trans.Resize({in2col_dims[0], in2col_dims[3], in2col_dims[1], in2col_dims[2], in2col_dims[4]});

        // convert input2col (S, B, C, filter_size * filter_size, H_output * W_output)
        // into input2col_trans (S, filter_size * filter_size, B, C, H_output * W_output)
        const int Rank = 5;
        Eigen::array<int, Rank>  permute;
        permute = {0, 3, 1, 2, 4};

        auto eigen_in = framework::EigenTensor<T, Rank>::From(input2col);
        auto eigen_out = framework::EigenTensor<T, Rank>::From(input2col_trans);
        auto* dev = dev_ctx.eigen_device();
        eigen_out.device(*dev) = eigen_in.shuffle(permute);

        permute = {0, 2, 3, 1, 4};

        if (pooling_type == "max") {
            // maxpooling
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->max_pooling(
                &input2col_trans, &max2col, &out_one_hot_tensor_trans);

            // convert out_one_hot_tensor_trans: (S, filter_size * filter_size, B, C, H_output * W_output)
            // into out_one_hot_tensor (S, B, C, filter_size * filter_size, H_output * W_output)
            auto eigen_in2 = framework::EigenTensor<T, Rank>::From(out_one_hot_tensor_trans);
            auto eigen_out2 = framework::EigenTensor<T, Rank>::From(*out_one_hot_tensor);
            eigen_out2.device(*dev) = eigen_in2.shuffle(permute);
        } else if (pooling_type == "avg") {
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->avg_pooling(
                &input2col_trans, &max2col);
        }

        // convert max2col: (S, 1, B, C, H_output * W_output)
        // into out_one_hot_tensor (S, B, C, 1, H_output * W_output)
        auto eigen_in3 = framework::EigenTensor<T, Rank>::From(max2col);

        // flatten height & width
        auto flatten_out_dims = out_dims;
        flatten_out_dims[3] = 1;
        flatten_out_dims[4] = out_dims[3] * out_dims[4];
        out->Resize(flatten_out_dims);

        auto eigen_out3 = framework::EigenTensor<T, Rank>::From(*out);
        eigen_out3.device(*dev) = eigen_in3.shuffle(permute);

        // reshape out (S, 1, B, C, H_output * W_output)
        // into (S, B, C, H_output * W_output)
        out->Resize(out_dims);
    }
};


template <typename DeviceContext, typename T>
class MpcPoolGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &context) const override {

        const Tensor* one_hot_tensor = context.Input<Tensor>("One_hot_tensor");
        const Tensor* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
        Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));

        std::string pooling_type = context.Attr<std::string>("pooling_type");
        std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
        std::vector<int> strides = context.Attr<std::vector<int>>("strides");
        std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
        std::string data_format = context.Attr<std::string>("data_format"); // NCHW
        bool global_pooling = context.Attr<bool>("global_pooling");
        std::string padding_algorithm =
            context.Attr<std::string>("padding_algorithm");

        if (in_x_grad) {
            // update padding => h, w
            auto in_x_dims = in_x_grad->dims();
            auto out_dims = out_grad->dims();
            framework::DDim data_dims;
            data_dims = framework::slice_ddim(in_x_dims, 3, in_x_dims.size());

            UpdatePadding(&paddings, global_pooling, padding_algorithm,
                          data_dims, strides, ksize);
            if (data_dims.size() * 2 == static_cast<int>(paddings.size())) {
              for (int i = 0; i < data_dims.size(); ++i) {
                paddings.erase(paddings.begin() + i + 1);
              }
            }

            if (global_pooling) {
              UpdateKsize(&ksize, data_dims);
            }

            // create temp tensor
            auto& dev_ctx = context.template device_context<DeviceContext>();
            Tensor expanded_out_grad_tensor =
                context.AllocateTmpTensor<T, DeviceContext>(one_hot_tensor->dims(), dev_ctx);
            Tensor mul_result_tensor =
                context.AllocateTmpTensor<T, DeviceContext>(one_hot_tensor->dims(), dev_ctx);

            // create data var of input and output variable
            T* in_x_grad_data = in_x_grad->mutable_data<T>(context.GetPlace());

            math::SetConstant<DeviceContext, T> set_constant;
            set_constant(dev_ctx, in_x_grad, 0);

            const T* one_hot_tensor_data = one_hot_tensor->data<T>();
            const T* out_grad_data = out_grad->data<T>();
            T* expanded_out_grad_data = expanded_out_grad_tensor.data<T>();
            T* mul_result_data = mul_result_tensor.data<T>();

            const int filter_numel = ksize[0] * ksize[1];

            // stride = h * w
            const int input_stride = in_x_dims[3] * in_x_dims[4];
            const int output_stride = out_dims[3] * out_dims[4];
            const int one_hot_tensor_stride = ksize[0] * ksize[1] * out_dims[3] * out_dims[4];

            // stride: share0, share1
            const int input_plaintext_size = in_x_grad->numel() / 2;
            const int output_plaintext_size = out_grad->numel() / 2;
            const int one_hot_tensor_plaintext_size = one_hot_tensor->numel() / 2;

            // expand out grad
            auto get_expand_out_grad =
                [filter_numel, one_hot_tensor_plaintext_size, output_plaintext_size]
#ifdef __NVCC__
            __host__ __device__
#endif
                (int ph, int pw, int input_height, int input_width,
                 int out_height, int out_width, int hstart, int hend,
                 int wstart, int wend, const T* src, T* target) {

                size_t out_grad_index = ph * out_width + pw;
                size_t offset = out_height * out_width;

                for (size_t index = 0; index < filter_numel; ++index) {
                    target[out_grad_index + index * offset] = src[out_grad_index]; //share0
                    target[out_grad_index + index * offset + one_hot_tensor_plaintext_size] =
                        src[out_grad_index + output_plaintext_size]; // share1
                }
            };

            // expand [S, B, C, H_poolout, W_poolout] into [S, B, C, ksize * ksize, H_poolout*W_poolout]
            auto visit_functor = VisitDataStrideWise<DeviceContext, T, decltype(get_expand_out_grad)>();

            visit_functor(in_x_dims, out_dims, ksize, strides, paddings, out_grad_data,
                          expanded_out_grad_data, output_stride, one_hot_tensor_stride, get_expand_out_grad);

            VLOG(3) << "pool type: " << pooling_type;
            // compute mul result = out_grad.expand * one_hot_tensor
            if (pooling_type == "max") {
                mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->arith_bool_mul(
                    &expanded_out_grad_tensor, one_hot_tensor, &mul_result_tensor);
            } else if (pooling_type == "avg") {
                float scale = 1.0 / (ksize[0] * ksize[1]);
                mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->scale(
                    &expanded_out_grad_tensor, scale, &mul_result_tensor);
            }

            // updata input X's grad
            auto update_in_grad = [input_plaintext_size, one_hot_tensor_plaintext_size]
#ifdef __NVCC__
            __host__ __device__
#endif
            (int ph, int pw, int input_height, int input_width, int out_height, int out_width,
             int hstart, int hend, int wstart, int wend, const T* src, T* target) {

                size_t index = 0;
                size_t in_pos = 0;
                size_t out_grad_index = ph * out_width + pw;
                size_t res_offset = out_height * out_width;
                for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                         in_pos = h * input_width + w;
                         target[in_pos] += src[out_grad_index + index * res_offset]; // share0
                         target[in_pos + input_plaintext_size] +=
                             src[out_grad_index + index * res_offset + one_hot_tensor_plaintext_size]; // share1
                         ++index;
                    }
                }
            };
            // convert [S, B, C, filter_size * filter_size, ] into [S, B, C, H, W]
            auto visit_functor_ = VisitDataStrideWise<DeviceContext, T, decltype(update_in_grad)>();
            visit_functor_(in_x_dims, out_dims, ksize, strides, paddings, mul_result_data,
                          in_x_grad_data, one_hot_tensor_stride, input_stride, update_in_grad);

        } //if (in_x_grad)
    } // void ComputeImpl
}; // class MpcPooliGradKernel

}  // namespace operators
}  // namespace paddle
