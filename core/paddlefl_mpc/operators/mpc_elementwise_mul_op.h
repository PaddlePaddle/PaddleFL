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

// This op is different with elementwise_add of PaddlePaddle.
// We only consider that the dimensions of X is equal with the dimensions of Y.

#pragma once
#include "mpc_op.h"
#include "paddle/fluid/platform/transform.h"
#include "core/paddlefl_mpc/operators/math/elementwise_op_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

const size_t SHARE_NUM =  2;

template <typename DeviceContext, typename T>
void Expand(const framework::LoDTensor* in_y_t, int axis, Tensor* y_expand_t, const framework::DDim &expand_dims, const framework::ExecutionContext &ctx) {

    T* y_expand_data = y_expand_t->mutable_data<T>(expand_dims, ctx.GetPlace());
    std::fill(y_expand_data, y_expand_data + y_expand_t->numel(), static_cast<T>(0));

    Tensor in_y_t_slice;
    Tensor y_expand_t_slice;

    for (size_t i = 0; i < SHARE_NUM; ++i) {
          y_expand_t_slice = y_expand_t->Slice(i, i + 1);
          in_y_t_slice = in_y_t->Slice(i, i + 1);

          auto y_expand_dims = y_expand_t_slice.dims();
          auto y_dims = in_y_t_slice.dims();

          axis = (axis == -1 ? y_expand_dims.size() - y_dims.size() : axis);

          PADDLE_ENFORCE(axis >= 0 && axis < y_expand_dims.size(),
                         "Axis should be in range [0, x_dims)");

          int pre, n, post;
          math::GetMidDims get_mid_dims;
          get_mid_dims(y_expand_dims, y_dims, axis, &pre, &n, &post);

          auto y_expand_ = y_expand_t_slice.data<T>();
          auto y_ = in_y_t_slice.data<T>();
          auto nx_ = y_expand_t_slice.numel();

          paddle::platform::Transform<DeviceContext> trans;
          if (post == 1) {
              trans(ctx.template device_context<DeviceContext>(), y_expand_, y_expand_ + nx_, 
                    math::RowwiseTransformIterator<T, DeviceContext>(y_, n),
                    y_expand_, math::AddFunctor<T>());
          } else {
              trans(ctx.template device_context<DeviceContext>(), y_expand_, y_expand_ + nx_, 
                    math::MidWiseTransformIterator<T, DeviceContext>(y_, n, post),
                    y_expand_, math::AddFunctor<T>());
          }
    }
}

template <typename DeviceContext, typename T>
class MpcElementwiseMulKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override{
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *out_t = ctx.Output<framework::LoDTensor>("Out");

        int axis = ctx.Attr<int>("axis");

        auto out = out_t->mutable_data<T>(ctx.GetPlace());

        if (in_x_t->dims() == in_y_t->dims()) {
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(in_x_t, in_y_t, out_t);
        } else {
            Tensor y_expand_t;
            // expand input in_y_t into y_expand_t (dims: in_x_t->dims)
            Expand<DeviceContext, T>(in_y_t, axis, &y_expand_t, in_x_t->dims(), ctx);
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(in_x_t, &y_expand_t, out_t);
        }
    }
};

template <typename DeviceContext, typename T>
class MpcElementwiseMulGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
        auto *dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
        auto *dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
        int axis = ctx.Attr<int>("axis");
        auto dout_data = dout->data<T>();

        if (dx && dy && (in_x_t->dims().size() == in_y_t->dims().size())) {
            dx->mutable_data<T>(ctx.GetPlace());
            dy->mutable_data<T>(ctx.GetPlace());
            // dx = dout * y
            // dy = dout * x
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(dout, in_y_t, dx);
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(dout, in_x_t, dy);
        }

        if (dx) {
            // dx = dout * y_expand
            auto dx_data = dx->mutable_data<T>(ctx.GetPlace());
            Tensor y_expand_t;
            // expand in_y_t into y_expand_t (in_x_t->dims)
            Expand<DeviceContext, T>(in_y_t, axis, &y_expand_t, in_x_t->dims(), ctx);
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(dout, &y_expand_t, dx);
        }

        if (dy) {
            // dy_expand = dout * x
            // dy = reduce(dy_expand)
            auto dy_data = dy->mutable_data<T>(ctx.GetPlace());
            Tensor dy_expand_t;
            T* dy_expand_t_data = dy_expand_t.mutable_data<T>(in_x_t->dims(), ctx.GetPlace());
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->mul(dout, in_x_t, &dy_expand_t);

            // reduce: dy_expand_t (dims: in_x_t->dims()) -> dy (dims: in_y_t->dims())
            auto x_dims = in_x_t->dims();
            auto y_dims = in_y_t->dims();

            axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
            PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                 "Axis should be in range [0, x_dims)");

            int pre, n, post;
            math::GetMidDims get_mid_dims;
            get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

            std::fill(dy_data, dy_data + dy->numel(), static_cast<T>(0));

            for (size_t i = 0; i < SHARE_NUM; ++i) {
                int y_offset = i * n;
                for (size_t j = 0; j < pre; ++j) {
                    for (size_t k = 0; k < n; ++k) {
                        for (size_t m = 0; m < post; ++m) {
                            int out_offset = i * pre * n * post + j * n * post + k * post + m;
                            dy_data[k + y_offset] += dy_expand_t_data[out_offset];
                        }
                    }
                }
            }
        }
    }
};

}  // namespace operators
}  // namespace paddle

