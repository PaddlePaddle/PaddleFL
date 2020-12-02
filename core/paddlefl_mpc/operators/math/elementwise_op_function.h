/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {
namespace math {

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

}  // namespace math
}  // namespace operators
}  // namespace paddle
