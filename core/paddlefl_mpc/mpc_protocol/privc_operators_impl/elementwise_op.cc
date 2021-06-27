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

// Description: implementations of elementwise_add_op according to privc protocol

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

void add(const Tensor *lhs, const Tensor *rhs, Tensor *out, int axis) {

        if (lhs->dims() == rhs->dims()) {
            PaddleTensor lhs_(ContextHolder::device_ctx(), *lhs);
            PaddleTensor rhs_(ContextHolder::device_ctx(), *rhs);
            PaddleTensor out_(ContextHolder::device_ctx(), *out);

            PrivCFixedTensor lhs_f(&lhs_);
            PrivCFixedTensor rhs_f(&rhs_);
            PrivCFixedTensor out_f(&out_);

            lhs_f.add(&rhs_f, &out_f);
        } else {
            auto x_dims = lhs->dims();
            auto y_dims = rhs->dims();

            axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);

            PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                            "Axis should be in range [0, x_dims)");

            int pre = 0, n = 0, post = 0;
            GetMidDims get_mid_dims;
            get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

            auto x_ = lhs->data<int64_t>();
            auto y_ = rhs->data<int64_t>();
            auto out_ = out->data<int64_t>();
            auto nx_ = lhs->numel();

            paddle::platform::Transform<CPUDeviceContext> trans;
            auto cpu_device_ctx = dynamic_cast<const CPUDeviceContext*>(ContextHolder::device_ctx());
            if (post == 1) {
                trans(*cpu_device_ctx, x_, x_ + nx_,
                    paddle::operators::math::RowwiseTransformIterator<int64_t, CPUDeviceContext>(y_, n),
                    out_, paddle::operators::math::AddFunctor<int64_t>());
            } else {
                trans(*cpu_device_ctx, x_, x_ + nx_,
                    paddle::operators::math::MidWiseTransformIterator<int64_t, CPUDeviceContext>(y_, n, post),
                    out_, paddle::operators::math::AddFunctor<int64_t>());
            }
        }
}

void add_grad(const Tensor *in_x_t, const Tensor *in_y_t, const Tensor *dout,
              Tensor *dx, Tensor *dy, int axis) {

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

                size_t share_num = 1;
                for (size_t i = 0; i < share_num; ++i) {
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

} // privc
} // operators
} // paddle
