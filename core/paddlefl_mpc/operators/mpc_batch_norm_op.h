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
#ifndef USE_CUDA
#include "./math/math_function.h"
#endif // USE_CUDA
#include "core/paddlefl_mpc/mpc_protocol/mpc_operators.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;
extern std::shared_ptr<mpc::MpcOperators> mpc_operators;
// TODO: remove dependency on aby3 protocol
const int MPC_ONE_SHARE = (1 << paddle::mpc::FIXED_POINTER_SCALING_FACTOR) / 3;

template <typename DeviceContext, typename T>
struct Expand {
    void operator()(const Tensor* input, Tensor* output, int S, int N, int C, int sample_size);

};

template <typename DeviceContext, typename T>
void TransToChannelFirst(const Tensor* input, Tensor* output, const framework::ExecutionContext &ctx) {
    // Transpose tensor
    // input shape: {S, N, C, H, W}
    // output shape: {C, S, N, H, W}
    // H and W is optional
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto input_dims = input->dims();
    switch (input_dims.size()) {
        case 3: {
            std::vector<int> axis{2, 0, 1};
            output->mutable_data<T>({input_dims[2], input_dims[0], input_dims[1]}, ctx.GetPlace());
            math::Transpose<DeviceContext, T, 3> trans3;
            trans3(dev_ctx, *input, output, axis);
            break;
        }
        case 4: {
            std::vector<int> axis{2, 0, 1, 3};
            output->mutable_data<T>({input_dims[2], input_dims[0], input_dims[1], input_dims[3]}, ctx.GetPlace());
            math::Transpose<DeviceContext, T, 4> trans4;
            trans4(dev_ctx, *input, output, axis);
            break;
        }
        case 5: {
            std::vector<int> axis{2, 0, 1, 3, 4};
            output->mutable_data<T>({input_dims[2], input_dims[0], input_dims[1], input_dims[3], input_dims[4]},
                                    ctx.GetPlace());
            math::Transpose<DeviceContext, T, 5> trans5;
            trans5(dev_ctx, *input, output, axis);
            break;
        }
        default:
            PADDLE_THROW("The size of input X's dimensions should be larger than 2, less than 6.");
    }
}

template <typename DeviceContext, typename T>
struct ComputeSum {
    void operator()(const Tensor* input,
                    int C,
                    Tensor* sum,
                    const framework::ExecutionContext &ctx);
};


template <typename DeviceContext, typename T>
void ComputeMeanVariance(const Tensor* input, int S, int N, int C, int sample_size,
                         Tensor* saved_mean_e, Tensor* saved_variance_e,
                         const framework::ExecutionContext &ctx) {
    // Compute mean and variance of each channel
    // input shape: {S, N, C, H, W}
    // output shape: {S, C}
    // H and W is optional
    VLOG(3) << "Compute the mean and variance of each channel";
    Tensor input_trans;
    TransToChannelFirst<DeviceContext, T>(input, &input_trans, ctx);

    auto compute_sum = ComputeSum<DeviceContext, T>();
    compute_sum(input, C, saved_mean_e, ctx);
    mpc_operators->scale(saved_mean_e, 1.0 / (N * sample_size), saved_mean_e); // scale

    Tensor saved_mean_e_expand;
    T* saved_mean_e_expand_data = saved_mean_e_expand.mutable_data<T>(input->dims(), ctx.GetPlace());

    auto expand_functor = Expand<DeviceContext, T>();
    expand_functor (saved_mean_e, &saved_mean_e_expand, S, N, C, sample_size);

    mpc_operators->sub(input, &saved_mean_e_expand, &saved_mean_e_expand);
    mpc_operators->elementwise_mul(&saved_mean_e_expand, &saved_mean_e_expand, &saved_mean_e_expand);
    compute_sum(&saved_mean_e_expand, C, saved_variance_e, ctx);
    mpc_operators->scale(saved_variance_e, 1.0 / (N * sample_size), saved_variance_e); // scale

}

template <typename DeviceContext, typename T>
class MpcBatchNormKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {

        mpc_operators = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
        VLOG(3) << "Start MpcBatchNormKernel.";
        const float epsilon = ctx.Attr<float>("epsilon");
        float momentum = ctx.Attr<float>("momentum");
        const bool is_test = ctx.Attr<bool>("is_test");
        const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
        const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
        bool test_mode = is_test && (!trainable_stats);

        bool global_stats = test_mode || use_global_stats;

        const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
        const DataLayout data_layout =
            framework::StringToDataLayout(data_layout_str);

        const Tensor *x = ctx.Input<Tensor>("X");
        const DDim x_dims = x->dims();
        PADDLE_ENFORCE_GE(
            x_dims.size(), 3,
            platform::errors::InvalidArgument(
                "The size of input X's dimensions should be larger than 2."
                "But received: the size of input X's dimensions is [%d]",
                x_dims.size()));
        PADDLE_ENFORCE_LE(
            x_dims.size(), 6,
            platform::errors::InvalidArgument(
                "The size of input X's dimensions should be less than 6."
                "But received: the size of input X's dimensionss is [%d]",
                x_dims.size()));

        const int S = 2; // share number
        const int N = x_dims[1];
        const int C = (data_layout == DataLayout::kNCHW ? x_dims[2] : x_dims[x_dims.size() - 1]);
        const int sample_size = x->numel() / S / N / C;

        auto *y = ctx.Output<Tensor>("Y");

        auto *mean_out = ctx.Output<Tensor>("MeanOut");
        auto *variance_out = ctx.Output<Tensor>("VarianceOut");
        auto *saved_mean = ctx.Output<Tensor>("SavedMean");
        auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

        // alloc memory
        y->mutable_data<T>(ctx.GetPlace());
        mean_out->mutable_data<T>(ctx.GetPlace());
        variance_out->mutable_data<T>(ctx.GetPlace());
        saved_mean->mutable_data<T>(ctx.GetPlace());
        saved_variance->mutable_data<T>(ctx.GetPlace());

        if (!global_stats) {
            if ((N * sample_size) == 1) {
                // Only 1 element in normalization dimension,
                // we skip the batch norm calculation, let y = x.
                framework::TensorCopy(*x, ctx.GetPlace(), y);
                return;
            }

            // saved_xx is use just in this batch of data
            // compute mean and variance
            switch (data_layout) {
                case DataLayout::kNCHW: {
                    ComputeMeanVariance<DeviceContext, T>(x, S, N, C, sample_size, saved_mean, saved_variance, ctx);
                    break;
                }
                default:
                    PADDLE_THROW("Unknown storage order: %s", data_layout_str);
            }

            // updata global mean and variance, for prediction
            if (ctx.HasInput("MomentumTensor")) {
                const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
                momentum = mom_tensor->data<float>()[0];
            }

            Tensor saved_mean_scale;
            Tensor mean_out_scale;
            saved_mean_scale.mutable_data<T>(saved_mean->dims(), ctx.GetPlace());
            mean_out_scale.mutable_data<T>(mean_out->dims(), ctx.GetPlace());

            mpc_operators->scale(mean_out, momentum, &mean_out_scale);
            mpc_operators->scale(saved_mean, 1.0 - momentum, &saved_mean_scale);
            mpc_operators->add(&mean_out_scale, &saved_mean_scale, mean_out);

            mpc_operators->scale(variance_out, momentum, &mean_out_scale);
            mpc_operators->scale(saved_variance, 1.0 - momentum, &saved_mean_scale);

            mpc_operators->add(&mean_out_scale, &saved_mean_scale, variance_out);
        }


        // use SavedMean and SavedVariance to do normalize
        // compute output y
        Tensor inv_std;
        Tensor mean_arr;
        inv_std.mutable_data<T>({S, C}, ctx.GetPlace());

        Tensor epsilon_expand;
        epsilon_expand.mutable_data<int64_t>({S, C}, ctx.GetPlace());

        math::SetConstant<DeviceContext, T> set_constant;
        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        set_constant(dev_ctx, &epsilon_expand, MPC_ONE_SHARE * epsilon); // TODO

        // inv_std = 1 / sqrt(variance + epsilon)
        if (global_stats) {
            const Tensor* variance = ctx.Input<Tensor>("Variance");
            Tensor var_plus_epsilon;
            var_plus_epsilon.mutable_data<T>({S, C}, ctx.GetPlace());

            mpc_operators->add(variance, &epsilon_expand, &var_plus_epsilon);
            mpc_operators->inverse_square_root(&var_plus_epsilon, &inv_std);

            mean_arr.ShareDataWith(*ctx.Input<Tensor>("Mean"));
        } else {
            Tensor var_plus_epsilon;
            var_plus_epsilon.mutable_data<T>({S, C}, ctx.GetPlace());
            mpc_operators->add(saved_variance, &epsilon_expand, &var_plus_epsilon);
            mpc_operators->inverse_square_root(&var_plus_epsilon, &inv_std);

            framework::TensorCopy(inv_std, ctx.GetPlace(), saved_variance);

            mean_arr.ShareDataWith(*saved_mean);
        }

        //   ((x - est_mean) * (inv_var) * scale + bias
        //   formula transform ====>
        //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
        const auto *scale = ctx.Input<Tensor>("Scale");
        const auto *bias = ctx.Input<Tensor>("Bias");

        Tensor new_scale;
        Tensor new_bias;
        Tensor new_bias_tmp;
        new_scale.mutable_data<T>(scale->dims(), ctx.GetPlace());
        new_bias.mutable_data<T>(scale->dims(), ctx.GetPlace());
        new_bias_tmp.mutable_data<T>(scale->dims(), ctx.GetPlace());

        mpc_operators->elementwise_mul(&inv_std, scale, &new_scale);
        mpc_operators->elementwise_mul(&mean_arr, &new_scale, &new_bias_tmp);
        mpc_operators->sub(bias, &new_bias_tmp, &new_bias);

        switch (data_layout) {
            case DataLayout::kNCHW: {
                Tensor x_new_scale;
                x_new_scale.mutable_data<T>(y->dims(), ctx.GetPlace());

                Tensor new_scale_expand;
                new_scale_expand.mutable_data<T>(x->dims(), ctx.GetPlace());

                auto expand_functor = Expand<DeviceContext, T>();
                expand_functor(&new_scale, &new_scale_expand, S, N, C, sample_size);

                Tensor new_bias_expand;
                new_bias_expand.mutable_data<T>(x->dims(), ctx.GetPlace());
                expand_functor(&new_bias, &new_bias_expand, S, N, C, sample_size);

                mpc_operators->elementwise_mul(x, &new_scale_expand, &x_new_scale);
                mpc_operators->add(&x_new_scale, &new_bias_expand, y);
                break;
            }
            default:
                PADDLE_THROW("Unknown storage order: %d", data_layout);
        }
    }
};


template <typename DeviceContext, typename T>
class MpcBatchNormGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {

        mpc_operators = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
        VLOG(3) << "Start MpcBatchNormGradKernel.";
        const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
        const auto *scale = ctx.Input<Tensor>("Scale");
        const auto *bias = ctx.Input<Tensor>("Bias");
        const auto *saved_mean = ctx.Input<Tensor>("SavedMean");

        // SavedVariance have been reverted in forward operator
        const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
        const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
        const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
        const bool is_test = ctx.Attr<bool>("is_test");
        const float epsilon = ctx.Attr<float>("epsilon");
        const DataLayout data_layout =
            framework::StringToDataLayout(data_layout_str);

        auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
        auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
        auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

        // batch_norm with inplace as false will take X as grad input, which
        // is same as cuDNN batch_norm backward calculation, batch_norm
        // with inplace as true only take Y as input and X should be calculate
        // by inverse operation of batch_norm on Y
        const Tensor *x;
        x = ctx.Input<Tensor>("X");

        PADDLE_ENFORCE_EQ(
            is_test, false,
            platform::errors::InvalidArgument(
                "`is_test = True` CANNOT be used in train program. If "
                "you want to use global status in pre_train model, "
                "please set `use_global_stats = True`"));

        // Get the size for each dimension.
        // NCHW [batch_size, in_channels, in_height, in_width]
        const auto &x_dims = x->dims();
        PADDLE_ENFORCE_GE(
            x_dims.size(), 3,
            platform::errors::InvalidArgument(
                "The size of input X's dimensions should be larger than 2."
                "But received: the size of input X's dimensions is [%d]",
                x_dims.size()));
        PADDLE_ENFORCE_LE(
            x_dims.size(), 6,
            platform::errors::InvalidArgument(
                "The size of input X's dimensions should be less than 6."
                "But received: the size of input X's dimensionss is [%d]",
                x_dims.size()));
        const int S = 2; // share number
        const int N = x_dims[1];
        const int C = (data_layout == DataLayout::kNCHW ? x_dims[2] : x_dims[x_dims.size() - 1]);
        const int sample_size = x->numel() / S / N / C;

        d_x->mutable_data<T>(ctx.GetPlace());

        const T *mean_data = saved_mean->data<T>();
        Tensor inv_var_tensor;
        inv_var_tensor.ShareDataWith(*saved_inv_variance); // local variance

        // update mean_data, compute inv_var = 1 / sqrt(variance + epsilon)
        if (use_global_stats) {
            const auto *running_mean = ctx.Input<Tensor>("Mean");
            const auto *running_variance = ctx.Input<Tensor>("Variance");
            mean_data = running_mean->data<T>();

            Tensor inv_var_tmp;
            inv_var_tmp.Resize({S, C});

            Tensor var_plus_epsilon;
            var_plus_epsilon.mutable_data<T>(running_variance->dims(), ctx.GetPlace());

            Tensor epsilon_expand;
            epsilon_expand.mutable_data<T>({S, C}, ctx.GetPlace());

            math::SetConstant<DeviceContext, T> set_constant;
            auto& dev_ctx = ctx.template device_context<DeviceContext>();
            set_constant(dev_ctx, &epsilon_expand, MPC_ONE_SHARE * epsilon); // TODO

            mpc_operators->add(running_variance, &epsilon_expand,  &var_plus_epsilon);
            mpc_operators->inverse_square_root(&var_plus_epsilon, &inv_var_tmp);
            framework::TensorCopy(inv_var_tmp, ctx.GetPlace(), &inv_var_tensor);
        }


        if (d_scale && d_bias) {
            d_scale->mutable_data<T>(ctx.GetPlace());
            d_bias->mutable_data<T>(ctx.GetPlace());
        }

        // d_bias = np.sum(d_y, axis=0)
        // d_scale = np.sum((X - mean) / inv_std * dy, axis=0)
        if ((N * sample_size) == 1 && !use_global_stats) {
            framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
            return;
        }


        switch (data_layout) {
            case DataLayout::kNCHW: {
                // d_bias = np.sum(d_y, axis=0)
                Tensor dy_sum;
                dy_sum.Resize({S, C});
                dy_sum.mutable_data<T>(ctx.GetPlace());

                auto compute_sum = ComputeSum<DeviceContext, T>();
                compute_sum(d_y, C, &dy_sum, ctx); // dy_sum

                // d_scale = np.sum((X - mean) / inv_std * dy, axis=0)
                // = [np.sum(X * dy) - mean * dy_sum] * inv_std
                Tensor x_mul_dy;
                x_mul_dy.mutable_data<T>(x->dims(), ctx.GetPlace());
                const DDim d_y_dim = d_y->dims();
                mpc_operators->elementwise_mul(x, d_y, &x_mul_dy); // X * dy

                Tensor dy_mul_x_sub_mean_mul_invstd_sum;
                dy_mul_x_sub_mean_mul_invstd_sum.mutable_data<T>({S, C}, ctx.GetPlace());
                compute_sum(&x_mul_dy, C, &dy_mul_x_sub_mean_mul_invstd_sum, ctx); // sum(X * dy)

                Tensor dy_sum_mul_mean;
                dy_sum_mul_mean.mutable_data<T>({S, C}, ctx.GetPlace());
                mpc_operators->elementwise_mul(&dy_sum, saved_mean, &dy_sum_mul_mean); // mean * dy_sum

                Tensor tmp;
                tmp.mutable_data<T>({S, C}, ctx.GetPlace());
                // [np.sum(X * dy) - mean * dy_sum]
                mpc_operators->sub(&dy_mul_x_sub_mean_mul_invstd_sum, &dy_sum_mul_mean, &tmp);
                // [np.sum(X * dy) - mean * dy_sum] * inv_std
                mpc_operators->elementwise_mul(&tmp, saved_inv_variance, &dy_mul_x_sub_mean_mul_invstd_sum);


                if (d_scale && d_bias) {
                    framework::TensorCopy(dy_sum, ctx.GetPlace(), d_bias);
                    framework::TensorCopy(dy_mul_x_sub_mean_mul_invstd_sum, ctx.GetPlace(), d_scale);
                }

                // d_x = (1. / N) * scale * inv_var * (N * d_y - np.sum(d_y, axis=0)
                // - (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))
                int scale_coefff = use_global_stats ? 1 : N * sample_size;

                Tensor scale_inv_var_nhw;
                T* scale_inv_var_nhw_data = scale_inv_var_nhw.mutable_data<T>({S, C}, ctx.GetPlace());
                // scale * inv_var
                mpc_operators->elementwise_mul(scale, saved_inv_variance, &scale_inv_var_nhw);
                // (1. / N) * scale * inv_var
                mpc_operators->scale(&scale_inv_var_nhw, 1.0 / scale_coefff, &scale_inv_var_nhw);
                Tensor scale_inv_var_nhw_expand;
                scale_inv_var_nhw_expand.mutable_data<T>(d_y_dim, ctx.GetPlace());

                auto expand_functor = Expand<DeviceContext, T>();
                expand_functor(&scale_inv_var_nhw, &scale_inv_var_nhw_expand, S, N, C, sample_size);

                if (!use_global_stats) {
                    Tensor dy_scale;
                    dy_scale.mutable_data<T>(d_y_dim, ctx.GetPlace());
                    // N * dy
                    mpc_operators->scale(d_y, N * sample_size, &dy_scale);

                    Tensor dy_sum_expand;
                    dy_sum_expand.mutable_data<T>(d_y_dim, ctx.GetPlace());
                    expand_functor(&dy_sum, &dy_sum_expand, S, N, C, sample_size);

                    Tensor dy_scale_minus_dy;
                    dy_scale_minus_dy.mutable_data<T>(d_y_dim, ctx.GetPlace());
                    // N * dy - np.sum(d_y, axis=0)
                    mpc_operators->sub(&dy_scale, &dy_sum_expand, &dy_scale_minus_dy);

                    Tensor mean_expand;
                    mean_expand.mutable_data<T>(d_y_dim, ctx.GetPlace());
                    expand_functor(saved_mean, &mean_expand, S, N, C, sample_size);

                    Tensor x_minus_mean;
                    x_minus_mean.mutable_data<T>(d_y_dim, ctx.GetPlace());
                    // (X - mean)
                    mpc_operators->sub(x, &mean_expand, &x_minus_mean);
                    //  inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))
                    mpc_operators->elementwise_mul(&dy_mul_x_sub_mean_mul_invstd_sum, saved_inv_variance, &tmp);

                    Tensor tmp_expand;
                    tmp_expand.mutable_data<T>(d_y_dim, ctx.GetPlace());
                    expand_functor(&tmp, &tmp_expand, S, N, C, sample_size);

                    Tensor tmp_expand2;
                    tmp_expand2.mutable_data<T>(d_y_dim, ctx.GetPlace());
                    // (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0)
                    mpc_operators->elementwise_mul(&tmp_expand, &x_minus_mean, &tmp_expand2);
                    mpc_operators->sub(&dy_scale_minus_dy, &tmp_expand2, &dy_scale);
                    mpc_operators->elementwise_mul(&scale_inv_var_nhw_expand, &dy_scale, d_x);
                } else {
                    mpc_operators->elementwise_mul(&scale_inv_var_nhw_expand, d_y, d_x);
                }
                break;
            }
            default:
                PADDLE_THROW("Unknown storage order: %s", data_layout_str);
        } // switch
    } // void ComputeImpl
}; // class MpcBatchNormGradKernel

}  // namespace operators
}  // namespace paddle
