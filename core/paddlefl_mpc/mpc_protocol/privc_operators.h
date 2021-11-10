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

// Description: implementations of all ops according to privc protocol

#pragma once

#include <utility>

#include "context_holder.h"
#include "mpc_operators.h"
#include "paddle/fluid/framework/tensor.h"
#include "core/privc/fixedpoint_tensor.h"
#include "core/privc/privc_context.h"
#include "core/common/paddle_tensor.h"
#include "core/paddlefl_mpc/mpc_protocol/privc_operators_impl/elementwise_op.h"
#include "core/paddlefl_mpc/mpc_protocol/privc_operators_impl/matrix_op.h"

namespace paddle {
namespace mpc {


using paddle::framework::Tensor;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;
using PrivCFixedTensor = privc::FixedPointTensor<int64_t, privc::PRIVC_FIXED_POINT_SCALING_FACTOR>;
using PaddleTensor = common::PaddleTensor<int64_t>;
namespace privc_op = paddle::operators::privc;


class PrivCOperatorsImpl : public MpcOperators {
public:
    void add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1) override {
        privc_op::add(lhs, rhs, out, axis);
    }

    // TODO: override
    void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PaddleTensor lhs_(device_ctx(), *lhs);
        PaddleTensor rhs_(device_ctx(), *rhs);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor lhs_f(&lhs_);
        PrivCFixedTensor rhs_f(&rhs_);
        PrivCFixedTensor out_f(&out_);

        lhs_f.sub(&rhs_f, &out_f);
    }

    void neg(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.negative(&out_f);
    }

    void sum(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.sum(&out_f);
    }

    // todo
    void elementwise_mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1) override {
        PaddleTensor lhs_(device_ctx(), *lhs);
        PaddleTensor rhs_(device_ctx(), *rhs);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor lhs_f(&lhs_);
        PrivCFixedTensor rhs_f(&rhs_);
        PrivCFixedTensor out_f(&out_);

        lhs_f.mul(&rhs_f, &out_f);
    }

    // todo
    void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
                bool trans_lhs = false, bool trans_rhs = false, bool sum_reduce_batch = false) override {

        if (sum_reduce_batch) {
            PADDLE_THROW(platform::errors::Unimplemented(
                    "sum reduce batch is not implemented."));
        }

        PaddleTensor lhs_(device_ctx(), *lhs);
        PaddleTensor rhs_(device_ctx(), *rhs);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor lhs_f(&lhs_);
        PrivCFixedTensor rhs_f(&rhs_);
        PrivCFixedTensor out_f(&out_);

        lhs_f.mat_mul(&rhs_f, &out_f);
    }

    void relu(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.relu(&out_f);
    }

    void sigmoid(const Tensor *op, Tensor *out, const std::string approx_mode) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.sigmoid(&out_f);
    }

    void softmax(const Tensor *op, Tensor *out, bool use_relu, bool use_long_div) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.softmax(&out_f, use_relu);
    }

    void argmax(const Tensor *op, Tensor *out) override {
        PaddleTensor op_(device_ctx(), *op);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor op_f(&op_);
        PrivCFixedTensor out_f(&out_);

        op_f.argmax(&out_f);
    }

    void scale(const Tensor *lhs, const double factor, Tensor *out) override {
        PaddleTensor lhs_(device_ctx(), *lhs);
        PaddleTensor out_(device_ctx(), *out);

        PrivCFixedTensor lhs_f(&lhs_);
        PrivCFixedTensor out_f(&out_);

        PaddleTensor scale_tensor(ContextHolder::device_ctx());
        scale_tensor.from_float_point_scalar(factor, lhs_f.shape(), privc::PRIVC_FIXED_POINT_SCALING_FACTOR);

        lhs_f.mul(&scale_tensor, &out_f);
    }

    void relu_with_derivative(const Tensor *op, Tensor *out,
                                      Tensor *derivative) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "relu_with_derivative is not implemented."));
    }

    void gt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "gt is not implemented."));
    }

    void geq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "geq is not implemented."));
    }

    void lt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "lt is not implemented."));
    }

    void leq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "leq is not implemented."));
    }

    void eq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "eq is not implemented."));
    }

    void neq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "neq is not implemented."));
    }

    void relu_grad(const Tensor *y, const Tensor *dy, Tensor *dx, const float point) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "relu_grad is not implemented."));
    }

    void arith_bool_mul(const Tensor* op_a, const Tensor* op_b, Tensor* out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "arith_bool_mul is not implemented."));
    }

    void max_pooling(const Tensor* in, Tensor* out, Tensor* pos_info) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "max_pooling is not implemented."));
    }

    void inverse_square_root(const Tensor* in, Tensor* out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "inverse_square_root is not implemented."));
    }

    void add_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout,
                  Tensor *dx, Tensor *dy, int axis = -1) override {
        privc_op::add_grad(lhs, rhs, dout, dx, dy, axis);
    }

    void elementwise_mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout,
                  Tensor *dx, Tensor *dy, int axis = -1) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "elementwise_mul_grad is not implemented."));
    }

    void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
                            int x_num_col_dims, int y_num_col_dims) override {
        privc_op::mul(lhs, rhs, out, x_num_col_dims, y_num_col_dims);
    }

    void mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout,
                                 Tensor *dx, Tensor *dy, int x_num_col_dims, int y_num_col_dims) override {
        privc_op::mul_grad(lhs, rhs, dout, dx, dy, x_num_col_dims, y_num_col_dims);
    }

    void mean(const Tensor *in, Tensor *out) override {
        double scale_factor = 1.0 / (in->numel());
        sum(in, out);
        scale(out, scale_factor, out);
    }

    void mean_grad(const Tensor *dout, Tensor *dx) override {
        auto dout_data = dout->data<int64_t>();
        auto dx_data = dx->data<int64_t>();
        int dx_size = dx->numel();
        for (size_t i = 0; i < dx_size; ++i) {
            dx_data[i] = dout_data[0];
        }

        double scale_factor = 1.0 / dx_size;
        scale(dx, scale_factor, dx);

    }

    void predicts_to_indices(const Tensor* in,
                                     Tensor* out,
                                     float threshold = 0.5) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "predicts_to_indices is not implemented."));
    }

    void calc_tp_fp_fn(const Tensor* indices,
                               const Tensor* labels,
                               Tensor* out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "calc_tp_fp_fn is not implemented."));
    }

    void calc_precision_recall(const Tensor* tp_fp_fn, Tensor* out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "calc_precision_recall is not implemented."));
    }

    void div(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "div is not implemented."));
    }

    // TODO
    void online_share(size_t party, const Tensor *input, Tensor *out) override {
        PADDLE_THROW(platform::errors::Unimplemented(
            "online share is not implemented."));
    }

private:

    static const paddle::platform::DeviceContext* device_ctx() {
        return ContextHolder::device_ctx();
    }

};

} // mpc
} // paddle
