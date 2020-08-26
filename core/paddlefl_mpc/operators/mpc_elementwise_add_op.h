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

template <typename T, typename DeviceContext>
class MidWiseTransformIterator;

template <typename T>
class MidWiseTransformIterator<T, platform::CPUDeviceContext>
    : public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t,
                           T *, T &> {
 public:
  MidWiseTransformIterator(const T *ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  MidWiseTransformIterator<T, platform::CPUDeviceContext> &operator++() {
    ++j_;
    if (UNLIKELY(j_ == post_)) {
      ++i_;
      j_ = 0;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }
    return *this;
  }

  MidWiseTransformIterator<T, platform::CPUDeviceContext> &operator+(int n) {
    while (n-- > 0) {
      ++j_;
      if (UNLIKELY(j_ == post_)) {
        ++i_;
        j_ = 0;
        if (UNLIKELY(i_ == n_)) {
          i_ = 0;
        }
      }
    }
    return *this;
  }

  bool operator==(const MidWiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const MidWiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int64_t i_;
  int64_t j_;
  int64_t n_;
  int64_t post_;
};

template <typename Functor, typename T, typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, const DeviceContext &ctx, Functor func,
                   const bool is_xsize_larger = true)
      : x_(x->data<T>()),
        y_(y->data<T>()),
        z_(z->mutable_data<OutType>(ctx.GetPlace())),
        nx_(x->numel()),
        ctx_(ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y->numel();
    }
  }

  inline void Run() const {
    platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_, x_, x_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(y_, n), z_, func_);
    } else {
      trans(ctx_, y_, y_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(x_, n), z_, func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_, x_, x_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(y_, n, post), z_, func_);
    } else {
      trans(ctx_, y_, y_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(x_, n, post), z_, func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const DeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
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

            auto x_ = in_x_t_slice.data<T>();
            auto y_ = in_y_t_slice.data<T>();
            auto out_ = out_t_slice.data<T>();
            auto nx_ = in_x_t_slice.numel();

            paddle::platform::Transform<DeviceContext> trans;
            if (post == 1) {
                trans(ctx.template device_context<DeviceContext>(), x_, x_ + nx_, 
                    RowwiseTransformIterator<T, DeviceContext>(y_, n),
                    out_, AddFunctor<T>());
            } else {
                trans(ctx.template device_context<DeviceContext>(), x_, x_ + nx_, 
                    MidWiseTransformIterator<T, DeviceContext>(y_, n, post),
                    out_, AddFunctor<T>());
            }
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

                std::fill(dy_data, dy_data + dy->numel(), static_cast<T>(0));

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
};

}  // namespace operators
}  // namespace paddle

