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

#include <map>

#include "mpc_operators.h"
#include "paddle/fluid/framework/tensor.h"
#include "core/privc3/fixedpoint_tensor.h"

namespace paddle {
namespace mpc {

const size_t ABY3_SCALING_FACTOR = FIXED_POINTER_SCALING_FACTOR;
using paddle::framework::Tensor;
using FixedTensor = aby3::FixedPointTensor<int64_t, ABY3_SCALING_FACTOR>;

class Aby3OperatorsImpl : public MpcOperators {
public:

    Aby3OperatorsImpl();

    void add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1) override;

    void add_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int axis = -1) override;

    void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void neg(const Tensor *op, Tensor *out) override;

    void sum(const Tensor *op, Tensor *out) override;

    void elementwise_mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1) override;

    void elementwise_mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int axis = -1) override;

    void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, int x_num_col_dims, int y_num_col_dims) override;

    void mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int x_num_col_dims, int y_num_col_dims) override;

    void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
                bool trans_lhs = false, bool trans_rhs = false,
                bool sum_reduce_batch = false) override;

    void mean(const Tensor *in, Tensor *out) override;

    void mean_grad(const Tensor *dout, Tensor *dx) override;

    void scale(const Tensor *lhs, const double factor, Tensor *out) override;

    void relu(const Tensor *op, Tensor *out) override;

    void relu_with_derivative(const Tensor *op, Tensor *out, Tensor *derivative) override;

    void sigmoid(const Tensor *op, Tensor *out, const std::string mode = "sigmoid") override;

    void softmax(const Tensor *op, Tensor *out, bool use_relu, bool use_long_div) override;

    void gt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void geq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void lt(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void leq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void eq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void neq(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void relu_grad(const Tensor *y, const Tensor *dy,
                   Tensor *dx, float point = 0.0f) override;

    void arith_bool_mul(const Tensor* op_a, const Tensor* op_b, Tensor* out) override;

    void max_pooling(const Tensor* in, Tensor* out, Tensor* pos_info) override;

    void avg_pooling(const Tensor* in, Tensor* out) override;

    void max(const Tensor* in, Tensor* out) override;

    void inverse_square_root(const Tensor* in, Tensor* out) override;

    // only support pred for 1 in binary classification for now
    void predicts_to_indices(const Tensor* in,
                             Tensor* out,
                             float threshold = 0.5) override;

    void calc_tp_fp_fn(const Tensor* indices,
                       const Tensor* labels,
                       Tensor* out) override;

    void calc_precision_recall(const Tensor* tp_fp_fn,
                               Tensor* out) override;

    void div(const Tensor *lhs, const Tensor *rhs, Tensor *out) override;

    void online_share(size_t party,
                      const Tensor *input,
                      Tensor *out) override;

    void reveal(const Tensor *in, Tensor* out) override;

    void argmax(const Tensor *op, Tensor *out) override;
private:

    typedef void(FixedTensor:: * sigmoid_func)(FixedTensor *fixed_tensor);

    std::map<std::string, sigmoid_func> sigmoid_func_map;

    void init_sigmoid_func_map();

};

} // mpc
} // paddle
