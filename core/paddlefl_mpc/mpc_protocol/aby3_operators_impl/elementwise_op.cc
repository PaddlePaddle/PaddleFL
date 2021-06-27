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

// Description: implementations of elementwise_add_op according to ABY3 protocol

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

void add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis) {

    PADDLE_ENFORCE(lhs->dims()[0] == 2 && rhs->dims()[0] == 2, 
        "The first dimension of input x of protocol ABY3 should be equal to 2.");

    if (lhs->dims() == rhs->dims()) {
        auto lhs_tuple = from_tensor(lhs);
        auto rhs_tuple = from_tensor(rhs);
        auto out_tuple = from_tensor(out);

        auto lhs_ = std::get<0>(lhs_tuple).get();
        auto rhs_ = std::get<0>(rhs_tuple).get();
        auto out_ = std::get<0>(out_tuple).get();

        lhs_->add(rhs_, out_);
    } else {
        Tensor in_x_t_slice;
        Tensor in_y_t_slice;
        Tensor out_t_slice;

        for (size_t i = 0; i < SHARE_NUM; ++i) {
            in_x_t_slice = lhs->Slice(i, i + 1);
            in_y_t_slice = rhs->Slice(i, i + 1);
            out_t_slice = out->Slice(i, i + 1);


            auto x_dims = in_x_t_slice.dims();
            auto y_dims = in_y_t_slice.dims();

            axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);

            PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                            "Axis should be in range [0, x_dims)");

            int pre=0, n=0, post=0;
            GetMidDims get_mid_dims;
            get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

            auto x_ = in_x_t_slice.data<int64_t>();
            auto y_ = in_y_t_slice.data<int64_t>();
            auto out_ = out_t_slice.data<int64_t>();
            auto nx_ = in_x_t_slice.numel();

            paddle::platform::Transform<CPUDeviceContext> trans;
            auto cpu_device_ctx = dynamic_cast<const CPUDeviceContext*>(ContextHolder::device_ctx());
            if (post == 1) {
                trans(*cpu_device_ctx, x_, x_ + nx_,
                    RowwiseTransformIterator<int64_t, CPUDeviceContext>(y_, n),
                    out_, AddFunctor<int64_t>());
            } else {
                trans(*cpu_device_ctx, x_, x_ + nx_,
                    MidWiseTransformIterator<int64_t, CPUDeviceContext>(y_, n, post),
                    out_, AddFunctor<int64_t>());
            }
        }
    }
}

void add_grad(const Tensor *in_x_t, const Tensor *in_y_t, const Tensor *dout, Tensor *dx, Tensor *dy, int axis) {
    auto ctx = ContextHolder::exec_ctx();
    auto dout_data = dout->data<int64_t>();
    if (dx) {
        auto dx_data = dx->mutable_data<int64_t>(ctx->GetPlace());
        for (size_t i = 0; i < dout->numel(); i++) {
            dx_data[i] = dout_data[i];
        }
    }

    if (dy) {
        auto dy_data = dy->mutable_data<int64_t>(ctx->GetPlace());
        if (in_x_t->dims().size() == in_y_t->dims().size()) {
            for (size_t i = 0; i < dout->numel(); i++) {
                dy_data[i] = dout_data[i];
            }
        } else {
            auto x_dims = in_x_t->dims();
            auto y_dims = in_y_t->dims();

            axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
            PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                 "Axis should be in range [0, x_dims)");

            int pre=0, n=0, post=0;
            GetMidDims get_mid_dims;
            get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

            std::fill(dy_data, dy_data + dy->numel(), static_cast<int64_t>(0));

            for (size_t i = 0; i < SHARE_NUM; ++i) {
                int y_offset = i * n;
                for (size_t j = 0; j < pre; ++j) {
                    for (size_t k = 0; k < n; ++k) {
                        for (size_t m = 0; m < post; ++m) {
                            int out_offset = i * pre * n * post + j * n * post + k * post + m;
                            dy_data[k + y_offset] += dout_data[out_offset];
                        }
                    }
                }
            }
        }
    }
}

void sub(const Tensor *lhs, const Tensor *rhs, Tensor *out) {
    auto lhs_tuple = from_tensor(lhs);
    auto rhs_tuple = from_tensor(rhs);
    auto out_tuple = from_tensor(out);

    auto lhs_ = std::get<0>(lhs_tuple).get();
    auto rhs_ = std::get<0>(rhs_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    lhs_->sub(rhs_, out_);
}

template <typename DeviceContext, typename T>
void Expand(const framework::Tensor* in_y_t, int axis, Tensor* y_expand_t, const framework::DDim &expand_dims, const framework::ExecutionContext *ctx) {

    T* y_expand_data = y_expand_t->mutable_data<T>(expand_dims, ctx->GetPlace());
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
          GetMidDims get_mid_dims;
          get_mid_dims(y_expand_dims, y_dims, axis, &pre, &n, &post);

          auto y_expand_ = y_expand_t_slice.data<T>();
          auto y_ = in_y_t_slice.data<T>();
          auto nx_ = y_expand_t_slice.numel();

          paddle::platform::Transform<DeviceContext> trans;
          if (post == 1) {
              trans(ctx->template device_context<DeviceContext>(), y_expand_, y_expand_ + nx_,
                    math::RowwiseTransformIterator<T, DeviceContext>(y_, n),
                    y_expand_, math::AddFunctor<T>());
          } else {
              trans(ctx->template device_context<DeviceContext>(), y_expand_, y_expand_ + nx_,
                    math::MidWiseTransformIterator<T, DeviceContext>(y_, n, post),
                    y_expand_, math::AddFunctor<T>());
          }
    }
}

void elementwise_mul(const Tensor *in_x_t, const Tensor *in_y_t, Tensor *out_t, int axis) {
    auto ctx = ContextHolder::exec_ctx();
    if (in_x_t->dims() == in_y_t->dims()) {
        elementwise_mul_with_same_dim(in_x_t, in_y_t, out_t);
    } else {
        Tensor y_expand_t;
        //expand input in_y_t into y_expand_t (dims: in_x_t->dims)
        Expand<CPUDeviceContext, int64_t>(in_y_t, axis, &y_expand_t, in_x_t->dims(), ctx);
        elementwise_mul_with_same_dim(in_x_t, &y_expand_t, out_t);
   }
}

void elementwise_mul_grad(const Tensor *in_x_t, const Tensor *in_y_t, const Tensor *dout, Tensor *dx, Tensor *dy, int axis) {
    auto ctx = ContextHolder::exec_ctx();
    if (dx) {
        // dx = dout * y_expand
        auto dx_data = dx->mutable_data<int64_t>(ctx->GetPlace());
        Tensor y_expand_t;
        // expand in_y_t into y_expand_t (in_x_t->dims)
        Expand<CPUDeviceContext, int64_t>(in_y_t, axis, &y_expand_t, in_x_t->dims(), ctx);
        elementwise_mul_with_same_dim(dout, &y_expand_t, dx);
    } 

    if (dy) {
        // dy_expand = dout * x
        // dy = reduce(dy_expand)
        auto dy_data = dy->mutable_data<int64_t>(ctx->GetPlace());
        Tensor dy_expand_t;
        int64_t* dy_expand_t_data = dy_expand_t.mutable_data<int64_t>(in_x_t->dims(), ctx->GetPlace());
        elementwise_mul_with_same_dim(dout, in_x_t, &dy_expand_t);
        // reduce: dy_expand_t (dims: in_x_t->dims()) -> dy (dims: in_y_t->dims())
        auto x_dims = in_x_t->dims();
        auto y_dims = in_y_t->dims();

        axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
        PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
             "Axis should be in range [0, x_dims)");

        int pre = 0, n = 0, post = 0;
        GetMidDims get_mid_dims;
        get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

        std::fill(dy_data, dy_data + dy->numel(), static_cast<int64_t>(0));

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

void elementwise_mul_with_same_dim(const Tensor *lhs, const Tensor *rhs, Tensor *out) {

    auto lhs_tuple = from_tensor(lhs);
    auto rhs_tuple = from_tensor(rhs);
    auto out_tuple = from_tensor(out);

    auto lhs_ = std::get<0>(lhs_tuple).get();
    auto rhs_ = std::get<0>(rhs_tuple).get();
    auto out_ = std::get<0>(out_tuple).get();

    lhs_->mul(rhs_, out_);
}

} // aby3
} // operators
} // paddle
