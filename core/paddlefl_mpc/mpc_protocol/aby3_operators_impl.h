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

#include "aby3_operators.h"

#include <utility>
#include <map>

#include "context_holder.h"
#include "paddle/fluid/framework/tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/privc3/aby3_context.h"
#include "core/privc3/fixedpoint_tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/matrix_op.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/elementwise_op.h"

#ifdef __NVCC__
#include "core/common/paddle_tensor_impl.cu.h"
#include "core/privc3/fixedpoint_tensor_imp.cu.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/common.cu.h"
#else // __NVCC__
#include "core/common/paddle_tensor.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/common.h"
#endif // __NVCC__

namespace paddle {
namespace mpc {

using aby3::ABY3Context;
// TODO: decide scaling factor
using BoolTensor = aby3::BooleanTensor<int64_t>;
namespace aby3_op = paddle::operators::aby3;

#ifdef __NVCC__
using PaddleTensor = common::CudaPaddleTensor<int64_t>;
#else // __NVCC__
using PaddleTensor = common::PaddleTensor<int64_t>;
#endif // __NVCC__

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

Aby3OperatorsImpl::Aby3OperatorsImpl() {
    init_sigmoid_func_map();
}

void Aby3OperatorsImpl::init_sigmoid_func_map() {
    sigmoid_func_map.insert(std::pair<std::string, sigmoid_func>("sigmoid",
        (sigmoid_func)(&FixedTensor::sigmoid)));
    sigmoid_func_map.insert(std::pair<std::string, sigmoid_func>("sigmoid_enhanced",
        (sigmoid_func)(&FixedTensor::sigmoid_enhanced)));
    sigmoid_func_map.insert(std::pair<std::string, sigmoid_func>("sigmoid_chebyshev",
        (sigmoid_func)(&FixedTensor::sigmoid_chebyshev)));
    sigmoid_func_map.insert(std::pair<std::string, sigmoid_func>("sigmoid_high_precision",
        (sigmoid_func)(&FixedTensor::sigmoid_high_precision)));
}

void Aby3OperatorsImpl::add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis) {
    aby3_op::add(lhs, rhs, out, axis);
}

void Aby3OperatorsImpl::add_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int axis) {
    aby3_op::add_grad(lhs, rhs, dout, dx, dy, axis);
}

void Aby3OperatorsImpl::sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) {
    aby3_op::sub(lhs, rhs, out);
}

void Aby3OperatorsImpl::neg(const Tensor *op, Tensor *out) {

    auto op_tuple = from_tensor(op);
    auto out_tuple = from_tensor(out);

    auto op_ = std::get<0>(op_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    op_->negative(out_);
}

void Aby3OperatorsImpl::sum(const Tensor *op, Tensor *out) {

    auto op_tuple = from_tensor(op);
    auto out_tuple = from_tensor(out);

    auto op_ = std::get<0>(op_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    op_->sum(out_);
}

void Aby3OperatorsImpl::elementwise_mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis) {
    aby3_op::elementwise_mul(lhs, rhs, out, axis);
}

void Aby3OperatorsImpl::elementwise_mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int axis) {
    aby3_op::elementwise_mul_grad(lhs, rhs, dout, dx, dy, axis);
}

void Aby3OperatorsImpl::mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, int x_num_col_dims, int y_num_col_dims) {
    aby3_op::mul(lhs, rhs, out, x_num_col_dims, y_num_col_dims);
}

void Aby3OperatorsImpl::mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int x_num_col_dims, int y_num_col_dims) {
    aby3_op::mul_grad(lhs, rhs, dout, dx, dy, x_num_col_dims, y_num_col_dims);
}

void Aby3OperatorsImpl::matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
            bool trans_lhs, bool trans_rhs, bool sum_reduce_batch) {

    auto lhs_tuple = from_tensor(lhs);
    auto rhs_tuple = from_tensor(rhs);
    auto out_tuple = from_tensor(out);

    auto lhs_ = std::get<0>(lhs_tuple).get();
    auto rhs_ = std::get<0>(rhs_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    lhs_->mat_mul(rhs_, out_, trans_lhs, trans_rhs, sum_reduce_batch);
}

void Aby3OperatorsImpl::mean(const Tensor *in, Tensor *out) {
    double scale_factor = 1.0 / (in->numel() / aby3_op::SHARE_NUM);
    sum(in, out);
    scale(out, scale_factor, out);
}

void Aby3OperatorsImpl::mean_grad(const Tensor *dout, Tensor *dx) {
    auto dout_data = dout->data<int64_t>();
    auto dx_data = dx->data<int64_t>();
    int dx_size = dx->numel() / aby3_op::SHARE_NUM;
    // for (size_t i = 0; i < dx_size; ++i) {
    //     dx_data[i] = dout_data[0];
    // }
    // for (size_t i = dx_size; i < dx->numel(); ++i) {
    //     dx_data[i] = dout_data[1];
    // }
#ifdef __NVCC__
    int64_t dout_data_[2];
    cudaMemcpy(dout_data_, dout_data, sizeof(dout_data), cudaMemcpyDeviceToHost);
#else // __NVCC__
    int64_t dout_data_[2] = { dout_data[0], dout_data[1] };
#endif // __NVCC__

    auto dx_ = from_tensor(dx);

    assign_to_tensor(std::get<1>(dx_).get(), dout_data_[0]);
    assign_to_tensor(std::get<2>(dx_).get(), dout_data_[0]);

    double scale_factor = 1.0 / dx_size;
    scale(dx, scale_factor, dx);
}

void Aby3OperatorsImpl::scale(const Tensor *lhs, const double factor, Tensor *out) {
    auto lhs_tuple = from_tensor(lhs);
    auto out_tuple = from_tensor(out);

    auto lhs_ = std::get<0>(lhs_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    PaddleTensor scale_tensor(ContextHolder::device_ctx());
    scale_tensor.from_float_point_scalar(factor, lhs_->shape(), ABY3_SCALING_FACTOR);

    lhs_->mul(&scale_tensor, out_);
}

void Aby3OperatorsImpl::relu(const Tensor *op, Tensor *out) {
    auto op_tuple = from_tensor(op);
    auto out_tuple = from_tensor(out);

    auto op_ = std::get<0>(op_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    op_->relu(out_);
}

void Aby3OperatorsImpl::relu_with_derivative(const Tensor *op, Tensor *out, Tensor *derivative) {
    auto op_tuple = from_tensor(op);
    auto out_tuple = from_tensor(out);
    auto der_tuple = from_tensor<BoolTensor>(derivative);

    auto op_ = std::get<0>(op_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();
    auto der_ = std::get<0>(der_tuple).get();

    op_->relu_with_derivative(out_, der_);
}

void Aby3OperatorsImpl::sigmoid(const Tensor *op, Tensor *out, const std::string mode) {

    auto op_tuple = from_tensor(op);
    auto out_tuple = from_tensor(out);

    auto op_ = std::get<0>(op_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    auto iter = sigmoid_func_map.find(mode);
    if(iter != sigmoid_func_map.end()) {
        (op_->*iter->second)(out_);
    } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "mode %s is not implemented.", mode));
    }
}

void Aby3OperatorsImpl::softmax(const Tensor *op, Tensor *out, bool use_relu, bool use_long_div) {
    auto op_tuple = from_tensor(op);
    auto out_tuple = from_tensor(out);

    auto op_ = std::get<0>(op_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    op_->softmax(out_, use_relu, use_long_div);
}

void Aby3OperatorsImpl::gt(const Tensor *lhs, const Tensor *rhs, Tensor *out) {

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

void Aby3OperatorsImpl::geq(const Tensor *lhs, const Tensor *rhs, Tensor *out) {
    auto lhs_tuple = from_tensor(lhs);

    auto lhs_ = std::get<0>(lhs_tuple).get();

    PaddleTensor rhs_(ContextHolder::device_ctx());
    rhs_.from_float_point_type<float>(*rhs, ABY3_SCALING_FACTOR);

    PaddleTensor out_(ContextHolder::device_ctx(), *out);

    auto tmp0 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());
    auto tmp1 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());

    BoolTensor bool_out(tmp0.get(), tmp1.get());

    lhs_->geq(&rhs_, &bool_out);

    bool_out.reveal(&out_);
}

void Aby3OperatorsImpl::lt(const Tensor *lhs, const Tensor *rhs, Tensor *out) {

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

void Aby3OperatorsImpl::leq(const Tensor *lhs, const Tensor *rhs, Tensor *out) {
    auto lhs_tuple = from_tensor(lhs);

    auto lhs_ = std::get<0>(lhs_tuple).get();

    PaddleTensor rhs_(ContextHolder::device_ctx(), *rhs);
    rhs_.from_float_point_type<float>(*rhs, ABY3_SCALING_FACTOR);

    PaddleTensor out_(ContextHolder::device_ctx(), *out);

    auto tmp0 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());
    auto tmp1 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());

    BoolTensor bool_out(tmp0.get(), tmp1.get());

    lhs_->leq(&rhs_, &bool_out);

    bool_out.reveal(&out_);
}

void Aby3OperatorsImpl::eq(const Tensor *lhs, const Tensor *rhs, Tensor *out) {

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

void Aby3OperatorsImpl::neq(const Tensor *lhs, const Tensor *rhs, Tensor *out) {
    auto lhs_tuple = from_tensor(lhs);

    auto lhs_ = std::get<0>(lhs_tuple).get();

    PaddleTensor rhs_(ContextHolder::device_ctx(), *rhs);
    rhs_.from_float_point_type<float>(*rhs, ABY3_SCALING_FACTOR);

    PaddleTensor out_(ContextHolder::device_ctx(), *out);

    auto tmp0 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());
    auto tmp1 = ContextHolder::tensor_factory()->create_int64_t(rhs_.shape());

    BoolTensor bool_out(tmp0.get(), tmp1.get());

    lhs_->neq(&rhs_, &bool_out);

    bool_out.reveal(&out_);
}

void Aby3OperatorsImpl::relu_grad(const Tensor *y, const Tensor *dy,
               Tensor *dx, float point) {

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

void Aby3OperatorsImpl::arith_bool_mul(const Tensor* op_a, const Tensor* op_b, Tensor* out) {

    auto a_tuple = from_tensor(op_a);
    auto a_ = std::get<0>(a_tuple).get();

    auto b_tuple = from_tensor<BoolTensor>(op_b);
    auto b_ = std::get<0>(b_tuple).get();

    auto out_tuple = from_tensor(out);
    auto out_ = std::get<0>(out_tuple).get();

    b_->mul(a_, out_);
}

void Aby3OperatorsImpl::max_pooling(const Tensor* in, Tensor* out, Tensor* pos_info) {

    auto a_tuple = from_tensor(in);
    auto a_ = std::get<0>(a_tuple).get();

    auto b_tuple = from_tensor<BoolTensor>(pos_info);
    auto b_ = std::get<0>(b_tuple).get();

    auto out_tuple = from_tensor(out);
    auto out_ = std::get<0>(out_tuple).get();

    a_->max_pooling(out_, b_);
}

void Aby3OperatorsImpl::avg_pooling(const Tensor* in, Tensor* out) {

    auto a_tuple = from_tensor(in);
    auto a_ = std::get<0>(a_tuple).get();

    auto out_tuple = from_tensor(out);
    auto out_ = std::get<0>(out_tuple).get();

    a_->avg_pooling(out_);
}

void Aby3OperatorsImpl::max(const Tensor* in, Tensor* out) {

    auto a_tuple = from_tensor(in);
    auto a_ = std::get<0>(a_tuple).get();

    auto out_tuple = from_tensor(out);
    auto out_ = std::get<0>(out_tuple).get();

    a_->max_pooling(out_, nullptr);
}

void Aby3OperatorsImpl::inverse_square_root(const Tensor* in, Tensor* out) {
    auto x_tuple = from_tensor(in);
    auto x_ = std::get<0>(x_tuple).get();

    auto y_tuple = from_tensor(out);
    auto y_ = std::get<0>(y_tuple).get();

    x_->inverse_square_root(y_);
}

// only support pred for 1 in binary classification for now
void Aby3OperatorsImpl::predicts_to_indices(const Tensor* in,
                         Tensor* out,
                         float threshold) {
    auto x_tuple = from_tensor(in);
    auto x_ = std::get<0>(x_tuple).get();

    auto y_tuple = from_tensor(out);
    auto y_ = std::get<0>(y_tuple).get();

    FixedTensor::preds_to_indices(x_, y_, threshold);
}

void Aby3OperatorsImpl::calc_tp_fp_fn(const Tensor* indices,
                   const Tensor* labels,
                   Tensor* out) {
    auto idx_tuple = from_tensor(indices);
    auto idx = std::get<0>(idx_tuple).get();

    auto lbl_tuple = from_tensor(labels);
    auto lbl = std::get<0>(lbl_tuple).get();

    auto out_tuple = from_tensor(out);
    auto out_ = std::get<0>(out_tuple).get();

    FixedTensor::calc_tp_fp_fn(idx, lbl, out_);
}

void Aby3OperatorsImpl::calc_precision_recall(const Tensor* tp_fp_fn,
                           Tensor* out) {
    auto in_tuple = from_tensor(tp_fp_fn);
    auto in = std::get<0>(in_tuple).get();

    PaddleTensor out_(ContextHolder::device_ctx(), *out);
    out_.scaling_factor() = ABY3_SCALING_FACTOR;

    FixedTensor::calc_precision_recall(in, &out_);
}

void Aby3OperatorsImpl::div(const Tensor *lhs, const Tensor *rhs, Tensor *out) {

    auto lhs_tuple = from_tensor(lhs);
    auto rhs_tuple = from_tensor(rhs);
    auto out_tuple = from_tensor(out);

    auto lhs_ = std::get<0>(lhs_tuple).get();
    auto rhs_ = std::get<0>(rhs_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    lhs_->long_div(rhs_, out_);

}
#ifdef __NVCC__

template <typename T0, typename T1>
__global__ void cu_cast(const T0* src, T1* dst, size_t numel, T1 scale) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    while (col < numel) {
        dst[col] = src[col] / scale;
        col += blockDim.x * gridDim.x;
    }
}

#endif // __NVCC__

template <typename T0, typename T1>
void cast_tensor(const Tensor *in, Tensor* out, T1 scale) {
    size_t numel = in->numel();
#ifdef __NVCC__
    dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
    dim3 grid_size = dim3((numel + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);

    cu_cast<T0, T1><<<grid_size, block_size, 0, mpc::AbstractContext::_s_stream>>>(
        in->data<T0>(), out->data<T1>(), numel, scale);
#else // __NVCC__
    std::transform(in->data<T0>(), in->data<T0>() + numel, out->data<T1>(),
                   [scale](T0 op) { return op / scale; });
#endif// __NVCC__
}

void Aby3OperatorsImpl::reveal(const Tensor *in, Tensor* out) {

    auto out_dims = framework::slice_ddim(in->dims(), 1, in->dims().size());
    out->mutable_data<float>(out_dims, ContextHolder::device_ctx()->GetPlace());

    Tensor temp;
    temp.mutable_data<int64_t>(out_dims, ContextHolder::device_ctx()->GetPlace());
    PaddleTensor temp0(ContextHolder::device_ctx(), temp);

    PaddleTensor out_(ContextHolder::device_ctx(), *out);

    auto in_tuple = from_tensor(in);
    auto in_ = std::get<0>(in_tuple).get();

    in_->reveal(&temp0);

    cast_tensor<int64_t, float>(&temp, out, std::pow(2.0, ABY3_SCALING_FACTOR));
}

void Aby3OperatorsImpl::argmax(const Tensor *op, Tensor *out) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "argmax is not implemented."));
}

void Aby3OperatorsImpl::online_share(size_t party,
                                     const Tensor *input,
                                     Tensor *out)  {
        PaddleTensor input_(ContextHolder::device_ctx(), *input);
        input_.from_float_point_type<float>(*input, ABY3_SCALING_FACTOR);

        auto out_tuple = from_tensor(out);
        auto out_ = std::get<0>(out_tuple).get();

        FixedTensor::online_share(party, &input_, out_);
}

} // mpc
} // paddle
