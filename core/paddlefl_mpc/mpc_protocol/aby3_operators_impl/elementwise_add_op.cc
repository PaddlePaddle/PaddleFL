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

#include <utility>

#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/elementwise_add_op.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators_impl/common.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators.h"
#include "core/paddlefl_mpc/operators/math/elementwise_op_function.h"

#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_operators.h"
#include "paddle/fluid/framework/tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/privc3/aby3_context.h"
#include "core/privc3/fixedpoint_tensor.h"
#include "core/privc3/boolean_tensor.h"
#include "core/common/paddle_tensor.h"

namespace paddle {
namespace operators {
namespace aby3 {

using namespace paddle::operators::math;
using namespace paddle::mpc;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;

void add_impl(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis) {

    PADDLE_ENFORCE(lhs->dims()[0] == 2, "The first dimension of input x should be equal to 2.");
    PADDLE_ENFORCE(rhs->dims()[0] == 2, "The first dimension of input y should be equal to 2.");

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

            int pre, n, post;
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

void add_grad_impl(const Tensor *in_x_t, const Tensor *in_y_t, const Tensor *dout, Tensor *dx, Tensor *dy, int axis) {
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

                int pre, n, post;
                math::GetMidDims get_mid_dims;
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

} // aby3
} // operators
} // paddle
