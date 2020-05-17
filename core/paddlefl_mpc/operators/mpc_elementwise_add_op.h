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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// paddle/fluid/operators/elementwise/elementwise_op_function.h
template <typename T, typename DeviceContext>
class RowwiseTransformIterator;

template <typename T>
class RowwiseTransformIterator<T, platform::CPUDeviceContext>
    : public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t, T *, T &> {
public:
    RowwiseTransformIterator(const T *ptr, int n) : ptr_(ptr), i_(0), n_(n) {}

    RowwiseTransformIterator<T, platform::CPUDeviceContext> &operator++() {
        ++i_;
        if (UNLIKELY(i_ == n_)) {
            i_ = 0;
        }
        return *this;
    }

    RowwiseTransformIterator<T, platform::CPUDeviceContext> &operator+(int n) {
        while (n-- > 0) {
            ++i_;
            if (UNLIKELY(i_ == n_)) {
                i_ = 0;
            }
        }

        return *this;
    }

    bool operator==(const RowwiseTransformIterator<T, platform::CPUDeviceContext> &rhs) const {
        return (ptr_ + i_) == &(*rhs);
    }

    bool operator!=(const RowwiseTransformIterator<T, platform::CPUDeviceContext> &rhs) const {
        return (ptr_ + i_) != &(*rhs);
    }

    const T &operator*() { return ptr_[i_]; }

private:
    const T *ptr_;
    int i_;
    int64_t n_;
};

template <typename T>
struct AddFunctor {
    inline HOSTDEVICE T operator()(T x, T y) { return x + y; }
};

struct GetMidDims {
    inline HOSTDEVICE void operator()(const framework::DDim &x_dims,
                         const framework::DDim &y_dims, const int axis,
                         int *pre, int *n, int *post)  {
        *pre = 1;
        *n = 1;
        *post = 1;
        for (int i = 1; i < axis + 1; ++i) {
            (*pre) *= x_dims[i];
        }

        for (int i = 1; i < y_dims.size(); ++i) {
            PADDLE_ENFORCE_EQ(x_dims[i + axis], y_dims[i],
                              "Broadcast dimension mismatch.");
            (*n) *= y_dims[i];
        }

        for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
            (*post) *= x_dims[i];
        }
    }
};

const size_t SHARE_NUM =  2;

template <typename DeviceContext, typename T>
class MpcElementwiseAddKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override{
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *out_t = ctx.Output<framework::LoDTensor>("Out");

        int axis = ctx.Attr<int>("axis");

        auto out = out_t->mutable_data<T>(ctx.GetPlace());

        if (in_x_t->dims() == in_y_t->dims()) {
            mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->add(in_x_t, in_y_t, out_t);
        } else {
            Tensor in_x_t_slice;
            Tensor in_y_t_slice;
            Tensor out_t_slice;

            for (size_t i = 0; i < SHARE_NUM; ++i) {
                in_x_t_slice = in_x_t->Slice(i, i + 1);
                in_y_t_slice = in_y_t->Slice(i, i + 1);
                out_t_slice = out_t->Slice(i, i + 1);

                auto x_dims = in_x_t_slice.dims();
                auto y_dims = in_y_t_slice.dims();

                axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
                PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(), 
                               "Axis should be in range [0, x_dims)");

                int pre, n, post;
                GetMidDims get_mid_dims;
                get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);
                PADDLE_ENFORCE_EQ(post, 1, 
                                  "post should be equal 1, but received post is [%s]", post);

                auto x_ = in_x_t_slice.data<T>();
                auto y_ = in_y_t_slice.data<T>();
                auto out_ = out_t_slice.data<T>();
                auto nx_ = in_x_t_slice.numel();
                paddle::platform::Transform<DeviceContext> trans;
                trans(ctx.template device_context<DeviceContext>(), x_, x_ + nx_, 
                     RowwiseTransformIterator<T, DeviceContext>(y_, n),
                     out_, AddFunctor<T>());
            }
        }
  }
};

template <typename DeviceContext, typename T>
class MpcElementwiseAddGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
        auto *dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
        auto *dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
        int axis = ctx.Attr<int>("axis");
        auto dout_data = dout->data<T>();

        if (dx) {
            auto dx_data = dx->mutable_data<T>(ctx.GetPlace());
            for (size_t i = 0; i < dout->numel(); i++) {
                dx_data[i] = dout_data[i];
            }
        }

        if (dy) {
            auto dy_data = dy->mutable_data<T>(ctx.GetPlace());
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
                GetMidDims get_mid_dims;
                get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);
                PADDLE_ENFORCE_EQ(post, 1, 
                                  "post should be equal 1, but received post is [%s]", post);

                for (size_t i = 0; i < SHARE_NUM; ++i) {
                    int y_offset = i * n;
                    for (size_t j = 0; j < pre; ++j) {
                        for (size_t k = 0; k < n; ++k) {
                            int out_offset = i * pre * n + j * n + k;
                            if (0 == j) {
                                dy_data[k + y_offset] = dout_data[out_offset];
                            } else {
                                dy_data[k + y_offset] += dout_data[out_offset];
                            }
                        }
                    }
                 }
            }
        }
    }
};

}  // namespace operators
}  // namespace paddle

