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
    virtual void add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1) = 0;

    virtual void add_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int axis = -1) = 0;

    virtual void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    virtual void neg(const Tensor *op, Tensor *out) = 0;

    virtual void sum(const Tensor *op, Tensor *out) = 0;

    /* This operator is used to perform multiplication.
    *  lhs's dimension should be equal to rhs's dimension.
    */ 
    virtual void elementwise_mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis = -1) = 0;

    virtual void elementwise_mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *dout, Tensor *dx, Tensor *dy, int axis = -1) = 0;

    /* This function is used to perform matrix multiplication.
     * The attribute `x_num_col_dims` and `y_num_col_dims` determines how $x$(lhs) and $y$(rhs) are flattened (Default: 1).
     *
     * If the input $x$ is a tensor with more than two dimensions, 
     * $x$ will be flattened into a two-dimensional matrix first. 
     * The flattening rule is: the first `num_col_dims` will be flattened to 
     * form the first dimension of the final matrix (the height of the matrix), 
     * and the rest `rank(x) - num_col_dims` dimensions are flattened to 
     * form the second dimension of the final matrix (the width of the matrix). 
     * As a result, height of the flattened matrix is equal to the product of $x$'s 
     * first `x_num_col_dims` dimensions' sizes, 
     * and width of the flattened matrix is equal to the product of $x$'s 
     * last `rank(x) - num_col_dims` dimensions' size.
     * 
     * see PaddlePaddle doc (API: mul) for details.
    */
    virtual void mul(const Tensor *lhs, const Tensor *rhs, Tensor *out, 
                            int x_num_col_dims = 1, int y_num_col_dims = 1) = 0;

    virtual void mul_grad(const Tensor *lhs, const Tensor *rhs, const Tensor *out, 
                                 Tensor *dx, Tensor *dy, int x_num_col_dims, int y_num_col_dims) = 0;
    
    /* This operator is used to perform (batched) matrix multiplication 
    *  over the last two dimensions of the input tensors $x$(lhs) and $y$(rhs).
    *  [Input]: The input tensors' rank can be 2 or 3.
    *  [trans_lhs] [trans_rhs]: Whether to transpose
    *
    *  Only following dims are supported:
    *  Mat A is [BatchSize, H, W] and Mat B is [BatchSize, H, W].
    *  Mat A is [BatchSize, H, W] and Mat B is [H, W].
    *  Mat A is [H, W] and Mat B is [H, W].
    *
    *  If a transpose flag is specified, the last two dimensions of the
    *  tensor are transposed. If the tensor is rank-1 of shape [D], then
    *  for $x$ it is treated as [1, D] in nontransposed form and as [D, 1]
    *  in transposed form, whereas for $y$ it is the opposite: It is treated
    *  as [D, 1] in nontransposed form and as [1, D] in transposed form.
    *
    */
    virtual void matmul(const Tensor *lhs,
                        const Tensor *rhs,
                        Tensor *out,
                        bool trans_lhs = false,
                        bool trans_rhs = false) = 0;

    virtual void mean(const Tensor *in, Tensor *out) = 0;

    virtual void mean_grad(const Tensor *dout, Tensor *dx) = 0;

    virtual void scale(const Tensor *lhs, const double factor, Tensor *out) = 0;

    virtual void relu(const Tensor *op, Tensor *out) = 0;

    virtual void relu_with_derivative(const Tensor *op, Tensor *out,
                                      Tensor *derivative) = 0;

    /* sigmoid function.
     * mode: sigmoid(piece_wise_3), sigmoid_enhanced(piece_wise_5), sigmoid_chebyshev, sigmoid_high_precision(exp)
     */ 
    virtual void sigmoid(const Tensor *op, Tensor *out, const std::string mode = "sigmoid") = 0;

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

    // column wise max
    // in shape [n, ...], out shape [1, ...]
    virtual void max(const Tensor* in, Tensor* out) {}

    virtual void inverse_square_root(const Tensor* in, Tensor* out) = 0;

    virtual void predicts_to_indices(const Tensor* in,
                                     Tensor* out,
                                     float threshold = 0.5) = 0;

    virtual void calc_tp_fp_fn(const Tensor* indices,
                               const Tensor* labels,
                               Tensor* out) = 0;

    virtual void calc_precision_recall(const Tensor* tp_fp_fn, Tensor* out) = 0;

    virtual void div(const Tensor *lhs, const Tensor *rhs, Tensor *out) = 0;

    // online reveal, only for debug
    // e.g.
    // Tensor tensor_to_print;
    // reveal(tensor_in, tensor_to_print);
    // std::cout << tensor_to_print;
    virtual void reveal(const Tensor *in, Tensor* out) {};

    // convert TensorAdapter to shares and distribute to all parties
    // party: the party who has original data.
    virtual void online_share(size_t party, const Tensor *input, Tensor *out) = 0;

    virtual void argmax(const Tensor *op, Tensor *out) = 0;
};

} // mpc
} // paddle

