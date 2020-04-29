// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "mpc_op.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_instance.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MpcMulKernel : public MpcOpKernel<T> {
public:
  void ComputeImpl(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Input<Tensor>("Y");
    auto *out = ctx.Output<Tensor>("Out");

    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    int x_mat_width = 1;
    int x_mat_height = 1;
    int y_mat_width = 1;
    int y_mat_height = 1;

    for (size_t i = 1; i < x_dims.size(); i++) {
      if (i <= x_num_col_dims) {
        x_mat_width *= x_dims[i];
      } else {
        x_mat_height *= x_dims[i];
      }
    }
    for (size_t i = 1; i < y_dims.size(); i++) {
      if (i <= y_num_col_dims) {
        x_mat_width *= y_dims[i];
      } else {
        y_mat_height *= y_dims[i];
      }
    }

    Tensor x_matrix;
    Tensor y_matrix;
    x_matrix.ShareDataWith(*x);
    y_matrix.ShareDataWith(*y);

    if (x_dims.size() > 3) {
      x_matrix.Resize({2, x_mat_width, x_mat_height});
    }

    if (y_dims.size() > 3) {
      y_matrix.Resize({2, y_mat_width, y_mat_height});
    }

    out->mutable_data<T>(ctx.GetPlace());

    auto out_dim = out->dims();
    if (out_dim.size() > 3) {
      out->Resize({2, x_mat_width, y_mat_height});
    }

    mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->matmul(
        &x_matrix, &y_matrix, out);

    if (out_dim.size() > 3) {
      out->Resize(out_dim);
    }
  }
};

template <typename DeviceContext, typename T>
class MpcMulGradKernel : public MpcOpKernel<T> {
public:
  void ComputeImpl(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto *y = ctx.Input<framework::LoDTensor>("Y");
    auto *dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    auto dout_dims = dout->dims();

    int x_mat_width = 1;
    int x_mat_height = 1;
    int y_mat_width = 1;
    int y_mat_height = 1;

    for (size_t i = 1; i < x_dims.size(); i++) {
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

    if (x_dims.size() > 3) {
      x_matrix.Resize({2, x_mat_width, x_mat_height});
    }

    if (y_dims.size() > 3) {
      y_matrix.Resize({2, y_mat_width, y_mat_height});
    }

    if (dout_dims.size() > 3) {
      dout_matrix.Resize({2, x_mat_width, y_mat_height});
    }

    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }

    Tensor x_matrix_trans;
    Tensor y_matrix_trans;
    x_matrix_trans.mutable_data<T>(x->dims(), ctx.GetPlace());
    y_matrix_trans.mutable_data<T>(y->dims(), ctx.GetPlace());

    if (x_dims.size() >= 3) {
      x_matrix_trans.Resize({2, x_mat_height, x_mat_width});
    }

    if (y_dims.size() >= 3) {
      y_matrix_trans.Resize({2, y_mat_height, y_mat_width});
    }

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    const int Rank = 3;

    Eigen::array<int, Rank> permute;
    permute[0] = 0;
    permute[1] = 2;
    permute[2] = 1;

    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      if (dx->dims().size() > 3) {
        dx->Resize({2, x_mat_width, x_mat_height});
      }
      auto eigen_in = framework::EigenTensor<T, Rank>::From(y_matrix);
      auto eigen_out = framework::EigenTensor<T, Rank>::From(y_matrix_trans);
      auto *dev = dev_ctx.eigen_device();
      eigen_out.device(*dev) = eigen_in.shuffle(permute);
      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->matmul(
          &dout_matrix, &y_matrix_trans, dx);
      auto dx_dim = dx->dims();
      if (dx_dim.size() > 3) {
        dx->Resize(dx_dim);
      }
    }

    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      if (dy->dims().size() > 3) {
        dy->Resize({2, y_mat_width, y_mat_height});
      }

      auto eigen_in = framework::EigenTensor<T, Rank>::From(x_matrix);
      auto eigen_out = framework::EigenTensor<T, Rank>::From(x_matrix_trans);
      auto *dev = dev_ctx.eigen_device();
      eigen_out.device(*dev) = eigen_in.shuffle(permute);
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->matmul(
          &x_matrix_trans, &dout_matrix, dy);
      auto dy_dim = dy->dims();
      if (dy_dim.size() > 3) {
        dy->Resize(dy_dim);
      }
    }
  }
};

} // namespace operators
} // namespace paddle
