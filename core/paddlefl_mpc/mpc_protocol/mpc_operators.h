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

// Description:
// abstract mpc operation interface

#pragma once

#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace mpc {

using paddle::framework::Tensor;

// TODO: decide scaling factor
const size_t FIXED_POINTER_SCALING_FACTOR = 16;

class MpcOperators {
public:
    virtual void add(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void neg(const Tensor *op, Tensor *out) = 0;

    virtual void sum(const Tensor *op, Tensor *out) = 0;

    virtual void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void scale(const Tensor *lhs, const double factor, Tensor *out) = 0;

    virtual void relu(const Tensor *op, Tensor *out) = 0;

    virtual void relu_with_derivative(const Tensor *op, Tensor *out,
                                      Tensor *derivative) = 0;

    virtual void sigmoid(const Tensor *op, Tensor *out) = 0;

    virtual void sigmoid_enhanced(const Tensor *op, Tensor *out) = 0;

    virtual void sigmoid_chebyshev(const Tensor *op, Tensor *out) = 0;

    virtual void softmax(const Tensor *op, Tensor *out, bool use_relu, bool use_long_div) = 0;

    virtual void gt(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void geq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void lt(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void leq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void eq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void neq(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void relu_grad(const Tensor *y, const Tensor *dy, Tensor *dx, const float point) = 0;

    // arithmetic tensor mult boolean tensor, element-wisely
    // see [ABY3, sec 5.4.1]
    // for aby3 only
    // example (in plaintext):
    // [1, 2, 3, 4] * [0, 0, 1, 0] = [0, 0, 3, 0]
    virtual void arith_bool_mul(const Tensor* op_a, const Tensor* op_b, Tensor* out) {}

    // max pooling in which shape of filter is nx1
    // pos_info keeps which element is max in a col, for backward grad
    // for filter in other shape, reshape input first
    virtual void max_pooling(const Tensor* in, Tensor* out, Tensor* pos_info) {}

    virtual void inverse_square_root(const Tensor* in, Tensor* out) = 0;

    virtual void predicts_to_indices(const Tensor* in,
                                     Tensor* out,
                                     float threshold = 0.5) = 0;

    virtual void calc_tp_fp_fn(const Tensor* indices,
                               const Tensor* labels,
                               Tensor* out) = 0;

    virtual void calc_precision_recall(const Tensor* tp_fp_fn, Tensor* out) = 0;

    virtual void div(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;
};

} // mpc
} // paddle

