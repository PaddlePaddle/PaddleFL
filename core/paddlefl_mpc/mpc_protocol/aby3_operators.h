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

// Description: implementations of each virtual op according to ABY3 protocol

#pragma once

#include <utility>

#include "context_holder.h"
#include "mpc_operators.h"
#include "paddle/fluid/framework/tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/privc3/aby3_context.h"
#include "core/privc3/fixedpoint_tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/privc3/paddle_tensor.h"

namespace paddle {
namespace mpc {

using paddle::framework::Tensor;
using aby3::ABY3Context;
// TODO: decide scaling factor
const size_t ABY3_SCALING_FACTOR = FIXED_POINTER_SCALING_FACTOR;
using FixedTensor = aby3::FixedPointTensor<int64_t, ABY3_SCALING_FACTOR>;
using BoolTensor = aby3::BooleanTensor<int64_t>;
using PaddleTensor = aby3::PaddleTensor<int64_t>;

class Aby3OperatorsImpl : public MpcOperators {
public:

    void add(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->add(rhs_, out_);

    }

    // TODO: override
    void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->sub(rhs_, out_);
    }

    void neg(const Tensor *op, Tensor *out) override {

        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        op_->negative(out_);
    }

    void sum(const Tensor *op, Tensor *out) override {

        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        op_->sum(out_);
    }

    void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->mul(rhs_, out_);
    }

    void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->mat_mul(rhs_, out_);
    }

    void scale(const Tensor *lhs, const double factor, Tensor *out) override {
        auto lhs_tuple = from_tensor(lhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        PaddleTensor scale_tensor(ContextHolder::device_ctx());
        scale_tensor.from_float_point_scalar(factor, lhs_->shape(), ABY3_SCALING_FACTOR);

        lhs_->mul(&scale_tensor, out_);
    }

    void relu(const Tensor *op, Tensor *out) override {
        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        op_->relu(out_);
    }

    void relu_with_derivative(const Tensor *op, Tensor *out, Tensor *derivative) override {
        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);
        auto der_tuple = from_tensor<BoolTensor>(derivative);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();
        auto der_ = std::get<0>(der_tuple).get();

        op_->relu_with_derivative(out_, der_);
    }

    void sigmoid(const Tensor *op, Tensor *out) override {
        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        op_->sigmoid(out_);
    }

    void sigmoid_enhanced(const Tensor *op, Tensor *out) override {
        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        op_->sigmoid_enhanced(out_);
    }

    void sigmoid_chebyshev(const Tensor *op, Tensor *out) override {
        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        op_->sigmoid_chebyshev(out_);
    }

    void softmax(const Tensor *op, Tensor *out, bool use_relu, bool use_long_div) override {
        auto op_tuple = from_tensor(op);
        auto out_tuple = from_tensor(out);

        auto op_ = std::get<0>(op_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        op_->softmax(out_, use_relu, use_long_div);
    }

    void gt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);

        auto lhs_ = std::get<0>(lhs_tuple).get();

        PaddleTensor rhs_(ContextHolder::device_ctx());
        rhs_.from_float_point_type<float>(*rhs, ABY3_SCALING_FACTOR);

        PaddleTensor out_(ContextHolder::device_ctx(), *out);

        auto tmp0 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());
        auto tmp1 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());

        BoolTensor bool_out(tmp0.get(), tmp1.get());

        lhs_->gt(&rhs_, &bool_out);

        bool_out.reveal(&out_);
    }

    void geq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        lt(lhs, rhs, out);
        std::transform(out->data<int64_t>(), out->data<int64_t>() + out->numel(),
                       out->data<int64_t>(), [](int64_t b) { return 1 - b; });
    }

    void lt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);

        auto lhs_ = std::get<0>(lhs_tuple).get();

        PaddleTensor rhs_(ContextHolder::device_ctx(), *rhs);
        rhs_.from_float_point_type<float>(*rhs, ABY3_SCALING_FACTOR);

        PaddleTensor out_(ContextHolder::device_ctx(), *out);

        auto tmp0 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());
        auto tmp1 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());

        BoolTensor bool_out(tmp0.get(), tmp1.get());

        lhs_->lt(&rhs_, &bool_out);

        bool_out.reveal(&out_);
    }

    void leq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        gt(lhs, rhs, out);
        std::transform(out->data<int64_t>(), out->data<int64_t>() + out->numel(),
                       out->data<int64_t>(), [](int64_t b) { return 1 - b; });
    }

    void eq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);

        auto lhs_ = std::get<0>(lhs_tuple).get();

        PaddleTensor rhs_(ContextHolder::device_ctx(), *rhs);
        rhs_.from_float_point_type<float>(*rhs, ABY3_SCALING_FACTOR);

        PaddleTensor out_(ContextHolder::device_ctx(), *out);

        auto tmp0 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());
        auto tmp1 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());

        BoolTensor bool_out(tmp0.get(), tmp1.get());

        lhs_->eq(&rhs_, &bool_out);

        bool_out.reveal(&out_);
    }

    void neq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {
        eq(lhs, rhs, out);
        std::transform(out->data<int64_t>(), out->data<int64_t>() + out->numel(),
                       out->data<int64_t>(), [](int64_t b) { return 1 - b; });
    }

    void relu_grad(const Tensor *y, const Tensor *dy,
                   Tensor *dx, float point = 0.0f) override {

        auto y_tuple = from_tensor(y);

        auto y_ = std::get<0>(y_tuple).get();

        PaddleTensor point_(ContextHolder::device_ctx());

        point_.from_float_point_scalar<float>(point, y_->shape(), ABY3_SCALING_FACTOR);

        auto tmp0 = ContextHolder::tensor_factory()->create_int64_t(y_->shape());
        auto tmp1 = ContextHolder::tensor_factory()->create_int64_t(y_->shape());

        BoolTensor bool_out(tmp0.get(), tmp1.get());

        y_->gt(&point_, &bool_out);

        auto out_tuple = from_tensor(dx);
        auto out_ = std::get<0>(out_tuple).get();

        auto dy_tuple = from_tensor(dy);
        auto dy_ = std::get<0>(dy_tuple).get();

        bool_out.mul(dy_, out_);
    }

    void arith_bool_mul(const Tensor* op_a, const Tensor* op_b, Tensor* out) override {

        auto a_tuple = from_tensor(op_a);
        auto a_ = std::get<0>(a_tuple).get();

        auto b_tuple = from_tensor<BoolTensor>(op_b);
        auto b_ = std::get<0>(b_tuple).get();

        auto out_tuple = from_tensor(out);
        auto out_ = std::get<0>(out_tuple).get();

        b_->mul(a_, out_);
    }

    void max_pooling(const Tensor* in, Tensor* out, Tensor* pos_info) override {

        auto a_tuple = from_tensor(in);
        auto a_ = std::get<0>(a_tuple).get();

        auto b_tuple = from_tensor<BoolTensor>(pos_info);
        auto b_ = std::get<0>(b_tuple).get();

        auto out_tuple = from_tensor(out);
        auto out_ = std::get<0>(out_tuple).get();

        a_->max_pooling(out_, b_);
    }

    void max(const Tensor* in, Tensor* out) override {

        auto a_tuple = from_tensor(in);
        auto a_ = std::get<0>(a_tuple).get();

        auto out_tuple = from_tensor(out);
        auto out_ = std::get<0>(out_tuple).get();

        a_->max_pooling(out_, nullptr);
    }

    void inverse_square_root(const Tensor* in, Tensor* out) override {
        auto x_tuple = from_tensor(in);
        auto x_ = std::get<0>(x_tuple).get();

        auto y_tuple = from_tensor(out);
        auto y_ = std::get<0>(y_tuple).get();

        x_->inverse_square_root(y_);
    }

    // only support pred for 1 in binary classification for now
    void predicts_to_indices(const Tensor* in,
                             Tensor* out,
                             float threshold = 0.5) override {
        auto x_tuple = from_tensor(in);
        auto x_ = std::get<0>(x_tuple).get();

        auto y_tuple = from_tensor(out);
        auto y_ = std::get<0>(y_tuple).get();

        FixedTensor::preds_to_indices(x_, y_, threshold);
    }

    void calc_tp_fp_fn(const Tensor* indices,
                       const Tensor* labels,
                       Tensor* out) override {
        auto idx_tuple = from_tensor(indices);
        auto idx = std::get<0>(idx_tuple).get();

        auto lbl_tuple = from_tensor(labels);
        auto lbl = std::get<0>(lbl_tuple).get();

        auto out_tuple = from_tensor(out);
        auto out_ = std::get<0>(out_tuple).get();

        FixedTensor::calc_tp_fp_fn(idx, lbl, out_);
    }

    void calc_precision_recall(const Tensor* tp_fp_fn,
                               Tensor* out) override {
        auto in_tuple = from_tensor(tp_fp_fn);
        auto in = std::get<0>(in_tuple).get();

        PaddleTensor out_(ContextHolder::device_ctx(), *out);
        out_.scaling_factor() = ABY3_SCALING_FACTOR;

        FixedTensor::calc_precision_recall(in, &out_);
    }

    void div(const Tensor *lhs, const Tensor *rhs, Tensor *out) override {

        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->long_div(rhs_, out_);

    }

private:
    template <typename T>
    std::tuple<
        std::shared_ptr<T>,
        std::shared_ptr<PaddleTensor>,
        std::shared_ptr<PaddleTensor> > from_tensor(const Tensor* t) {

            PADDLE_ENFORCE_EQ(t->dims()[0], 2);

            auto pt0 = std::make_shared<PaddleTensor>(ContextHolder::device_ctx(), t->Slice(0, 1));
            auto pt1 = std::make_shared<PaddleTensor>(ContextHolder::device_ctx(), t->Slice(1, 2));

           // remove leading 1 in shape
           auto shape = pt0->shape();
           shape.erase(shape.begin());
           pt0->reshape(shape);
           pt1->reshape(shape);

            aby3::TensorAdapter<int64_t>* pt_array[2] = {pt0.get(), pt1.get()};

            auto ft = std::make_shared<T>(pt_array);

        return std::make_tuple(ft, pt0, pt1);
    }

    std::tuple<
        std::shared_ptr<FixedTensor>,
        std::shared_ptr<PaddleTensor>,
        std::shared_ptr<PaddleTensor> > from_tensor(const Tensor* t) {

        return from_tensor<FixedTensor>(t);
    }

};

} // mpc
} // paddle
