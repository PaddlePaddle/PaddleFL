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

// Description: implementations of mul_op according to privc protocol

#include "paddle/fluid/framework/tensor.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/paddlefl_mpc/operators/math/elementwise_op_function.h"
#include "core/paddlefl_mpc/mpc_protocol/privc_operators_impl/elementwise_op.h"
#include "core/paddlefl_mpc/mpc_protocol/privc_operators_impl/common.h"
#include "core/privc/fixedpoint_tensor.h"
#include "core/privc/privc_context.h"
#include "core/common/paddle_tensor.h"

namespace paddle {
namespace operators {
namespace privc {

using paddle::framework::Tensor;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;
using ::privc::PrivCContext;
using paddle::mpc::ContextHolder;
using PaddleTensor = common::PaddleTensor<int64_t>;
using PrivCFixedTensor = ::privc::FixedPointTensor<int64_t, ::privc::PRIVC_FIXED_POINT_SCALING_FACTOR>;
using paddle::mpc::ContextHolder;

void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
            bool trans_lhs = false, bool trans_rhs = false) {
    PaddleTensor lhs_(ContextHolder::device_ctx(), *lhs);
    PaddleTensor rhs_(ContextHolder::device_ctx(), *rhs);
    PaddleTensor out_(ContextHolder::device_ctx(), *out);

    PrivCFixedTensor lhs_f(&lhs_);
    PrivCFixedTensor rhs_f(&rhs_);
    PrivCFixedTensor out_f(&out_);

    lhs_f.mat_mul(&rhs_f, &out_f);
}

void mul(const Tensor *x, const Tensor *y, Tensor *out,
         int x_num_col_dims, int y_num_col_dims) {

    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, ContextHolder::exec_ctx()->template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, ContextHolder::exec_ctx()->template Attr<int>("y_num_col_dims"))
            : *y;

    auto out_dim = out->dims();
    if (out_dim.size() != 2) {
        out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    matmul(&x_matrix, &y_matrix, out);

    if (out_dim.size() != 2) {
        out->Resize(out_dim);
    }

}


void mul_grad(const Tensor *x, const Tensor *y, const Tensor *dout,
              Tensor *dx, Tensor *dy, int x_num_col_dims, int y_num_col_dims) {
        
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    auto dout_dims = dout->dims();

    auto x_matrix = x->dims().size() > 2
                        ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                        : static_cast<const Tensor&>(*x);
    auto y_matrix = y->dims().size() > 2
                        ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                            : static_cast<const Tensor&>(*y);

    Tensor dout_mat;
    dout_mat.ShareDataWith(*dout);
    dout_mat.Resize({framework::flatten_to_2d(x->dims(), x_num_col_dims)[0],
                     framework::flatten_to_2d(y->dims(), y_num_col_dims)[1]});


    auto transpose2 = [](const Tensor* in, Tensor* out) {
            operators::math::Transpose<CPUDeviceContext, int64_t, 2> trans2;
            std::vector<int> axis({1, 0});
            trans2(*dynamic_cast<const CPUDeviceContext*>(ContextHolder::device_ctx()), *in, out, axis);
    };

    PaddleTensor dout_mat_(ContextHolder::device_ctx(), dout_mat);
    PrivCFixedTensor dout_mat_f(&dout_mat_);

    if (dx) {
        Tensor dx_matrix = dx->dims().size() > 2
                                ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                                : *dx;
        // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
        // blas.MatMul(dout_mat, false, y_matrix, true, &dx_matrix);
        Tensor y_matrix_t;
        y_matrix_t.mutable_data<int64_t>({y_matrix.dims()[1], y_matrix.dims()[0]}, ContextHolder::exec_ctx()->GetPlace());
        transpose2(&y_matrix, &y_matrix_t);

        PaddleTensor y_matrix_(ContextHolder::device_ctx(), y_matrix_t);
        PaddleTensor dx_matrix_(ContextHolder::device_ctx(), dx_matrix);

        PrivCFixedTensor y_matrix_f(&y_matrix_);
        PrivCFixedTensor dx_matrix_f(&dx_matrix_);

        dout_mat_f.mat_mul(&y_matrix_f, &dx_matrix_f);
    }

    if (dy) {
        Tensor dy_matrix = dy->dims().size() > 2
                                ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                                : *dy;
        // dy = x' * dout. dy K x N, dout : M x N, x : M x K
        // blas.MatMul(x_matrix, true, dout_mat, false, &dy_matrix);
        Tensor x_matrix_t;
        x_matrix_t.mutable_data<int64_t>({x_matrix.dims()[1], x_matrix.dims()[0]}, ContextHolder::exec_ctx()->GetPlace());
        transpose2(&x_matrix, &x_matrix_t);

        PaddleTensor x_matrix_(ContextHolder::device_ctx(), x_matrix_t);
        PaddleTensor dy_matrix_(ContextHolder::device_ctx(), dy_matrix);

        PrivCFixedTensor x_matrix_f(&x_matrix_);
        PrivCFixedTensor dy_matrix_f(&dy_matrix_);

        x_matrix_f.mat_mul(&dout_mat_f, &dy_matrix_f);
    }
}


} // privc
} // operators
} // paddle
