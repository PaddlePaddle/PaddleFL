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

// Description: implementations of matrix ops(mul) according to ABY3 protocol

#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/common.h"
#include "core/paddlefl_mpc/operators/math/elementwise_op_function.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/elementwise_op.h"

namespace paddle {
namespace operators {
namespace aby3 {

using paddle::framework::Tensor;
using namespace paddle::operators::math;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;
using ::aby3::ABY3Context;
using paddle::mpc::ContextHolder;

void matmul(const Tensor *lhs, const Tensor *rhs, Tensor *out,
                bool trans_lhs = false, bool trans_rhs = false) {

        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->mat_mul(rhs_, out_, trans_lhs, trans_rhs);
    }

void mul(const Tensor *x, const Tensor *y, Tensor *out, int x_num_col_dims, int y_num_col_dims) {

    auto x_dims = x->dims();
    auto y_dims = y->dims();

    int x_mat_width = 1;
    int x_mat_height = 1;
    int y_mat_width = 1;
    int y_mat_height = 1;

    int x_batch_size = x_dims[1];

    for (size_t i = 2; i < x_dims.size(); i++) {
        if (i <= x_num_col_dims) {
            x_mat_width *= x_dims[i];
        } else {
            x_mat_height *= x_dims[i];
        }
    }
    for (size_t i = 1; i < y_dims.size(); i++) {
        if (i <= y_num_col_dims) {
            y_mat_width *= y_dims[i];
        } else {
            y_mat_height *= y_dims[i];
        }
    }

    Tensor x_matrix;
    Tensor y_matrix;
    x_matrix.ShareDataWith(*x);
    y_matrix.ShareDataWith(*y);

    x_matrix.Resize({SHARE_NUM, x_batch_size, x_mat_width, x_mat_height});
    y_matrix.Resize({SHARE_NUM, y_mat_width, y_mat_height});

    auto out_dim = out->dims();
    out->Resize({SHARE_NUM, x_batch_size, x_mat_width, y_mat_height});

    matmul(&x_matrix, &y_matrix, out);

    out->Resize(out_dim);

}

void mul_grad(const Tensor *x, const Tensor *y, const Tensor *dout, Tensor *dx, Tensor *dy, int x_num_col_dims, int y_num_col_dims) {

        auto x_dims = x->dims();
        auto y_dims = y->dims();
        auto dout_dims = dout->dims();

        int x_mat_width = 1;
        int x_mat_height = 1;
        int y_mat_width = 1;
        int y_mat_height = 1;

        int x_batch_size = x_dims[1];

        for (size_t i = 2; i < x_dims.size(); i++) {
            if (i <= x_num_col_dims) {
                x_mat_width *= x_dims[i];
            } else {
                x_mat_height *= x_dims[i];
            }
        }
        for (size_t i = 1; i < y_dims.size(); i++) {
            if (i <= y_num_col_dims) {
                y_mat_width *= y_dims[i];
            } else {
                y_mat_height *= y_dims[i];
            }
        }
        Tensor x_matrix;
        Tensor y_matrix;
        Tensor dout_matrix;
        x_matrix.ShareDataWith(*x);
        y_matrix.ShareDataWith(*y);
        dout_matrix.ShareDataWith(*dout);

        x_matrix.Resize({SHARE_NUM, x_batch_size, x_mat_width, x_mat_height});
        y_matrix.Resize({SHARE_NUM, y_mat_width, y_mat_height});
        dout_matrix.Resize({SHARE_NUM, x_batch_size, x_mat_width, y_mat_height});

        auto dev_ctx = dynamic_cast<const CPUDeviceContext*>(ContextHolder::device_ctx());

        if (dx) {
            auto dx_dim = dx->dims();
            dx->Resize({SHARE_NUM, x_batch_size, x_mat_width, x_mat_height});
            // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
            matmul(&dout_matrix, &y_matrix, dx, 0, 1);
            dx->Resize(dx_dim);
        }

        if (dy) {
            auto dy_dim = dy->dims();
            dy->Resize({2, y_mat_width, y_mat_height});

            // dy = x' * dout. dy K x N, dout : M x N, x : M x K
            x_matrix.Resize({2, x_batch_size * x_mat_width, x_mat_height});
            dout_matrix.Resize({2, x_batch_size * x_mat_width, y_mat_height});
            matmul(&x_matrix, &dout_matrix, dy, 1, 0);
            dy->Resize(dy_dim);
        }

}

} // aby3
} // operators
} // paddle
