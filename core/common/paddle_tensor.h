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

#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

#ifndef __NVCC__
#include "core/paddlefl_mpc/operators/math/math_function.h"
#else
#include "paddle/fluid/operators/math/math_function.h"
#endif

#include "tensor_adapter.h"
#include "tensor_adapter_factory.h"

namespace common {

template <typename T>
class PaddleTensor : public TensorAdapter<T> {
public:
  PaddleTensor(const paddle::platform::DeviceContext *device_ctx)
      : _device_ctx(device_ctx), _scaling_factor(0) {
    _device_ctx = device_ctx;
  }

  PaddleTensor(const paddle::platform::DeviceContext *device_ctx,
               const paddle::framework::Tensor &src)
      : PaddleTensor(_device_ctx) {
    // it seems that init list failed, weird
    _device_ctx = device_ctx;
    _tensor.ShareDataWith(src);
  }

  virtual ~PaddleTensor() = default;

  T *data() override { return _tensor.data<T>(); }

  const T *data() const override { return _tensor.data<T>(); }

  const paddle::framework::Tensor* paddle_tensor() const {
    return &_tensor;
  }

  paddle::framework::Tensor* mutable_paddle_tensor() {
    return &_tensor;
  }

  std::vector<size_t> shape() const override {
    return paddle::framework::vectorize<size_t>(_tensor.dims());
  }

  size_t numel() const override { return _tensor.numel(); }

  void add(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void sum(TensorAdapter<T> *ret) const override;

  void sub(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void negative(TensorAdapter<T> *ret) const override;

  void mul(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void div(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void mat_mul(const TensorAdapter<T> *rhs,
               TensorAdapter<T> *ret,
               bool transpose_lhs = false,
               bool transpose_rhs = false,
               bool sum_reduce_batch = false) const override;

  void add128(const TensorAdapter<T> *rhs,
              TensorAdapter<T> *ret,
              bool lhs_128,
              bool rhs_128) const override;

  void sub128(const TensorAdapter<T> *rhs,
              TensorAdapter<T> *ret,
              bool lhs_128,
              bool rhs_128) const override;

  void mul128_with_truncate(const TensorAdapter<T> *rhs,
                            TensorAdapter<T> *ret,
                            bool lhs_128,
                            bool rhs_128) const override;

  void bitwise_xor(const TensorAdapter<T> *rhs,
                   TensorAdapter<T> *ret) const override;

  void bitwise_and(const TensorAdapter<T> *rhs,
                   TensorAdapter<T> *ret) const override;

  void bitwise_or(const TensorAdapter<T> *rhs,
                  TensorAdapter<T> *ret) const override;

  void bitwise_not(TensorAdapter<T> *ret) const override;

  void lshift(size_t rhs, TensorAdapter<T> *ret) const override;

  void rshift(size_t rhs, TensorAdapter<T> *ret) const override;

  void logical_rshift(size_t rhs, TensorAdapter<T> *ret) const override;

  paddle::framework::Tensor &tensor() { return _tensor; }

  const paddle::framework::Tensor &tensor() const { return _tensor; }

  void reshape(const std::vector<size_t> &shape);

  const paddle::platform::DeviceContext *device_ctx() const {
    return _device_ctx;
  }

  size_t scaling_factor() const override { return _scaling_factor; }

  size_t &scaling_factor() override { return _scaling_factor; }

  void slice(size_t begin_idx, size_t end_idx,
             TensorAdapter<T> *ret) const override;

  template <typename U>
  PaddleTensor &from_float_point_type(const paddle::framework::Tensor &tensor,
                                      size_t scaling_factor);

  template <typename U>
  PaddleTensor &from_float_point_scalar(const U &scalar,
                                        const std::vector<size_t> &shape,
                                        size_t scaling_factor);

  template<int Rank>
  void Transpose(const std::vector<int> axis, TensorAdapter<T>* ret) const {
    paddle::operators::math::Transpose<paddle::platform::CPUDeviceContext, T, Rank> trans;
    trans(*(dynamic_cast<const paddle::platform::CPUDeviceContext*>(_device_ctx)),
          _tensor,
          dynamic_cast<PaddleTensor<T>*>(ret)->mutable_paddle_tensor(),
          axis);
  }

  template<int Rank>
  void Broadcast(const std::vector<int> axis, TensorAdapter<T>* ret) const {
    // input, output and axis rank = Rank
    // TODO: arbitrary ranks
    auto in = paddle::framework::EigenTensor<T, Rank>::From(_tensor);
    auto out = paddle::framework::EigenTensor<T, Rank>::From(dynamic_cast<PaddleTensor<T>*>(ret)->_tensor);

    Eigen::array<int, Rank> bcast;
    for (int i = 0; i < Rank; i++) {
        bcast[i] = axis[i];
    }

    auto &place = *eigen_device();
    out.device(place) = in.broadcast(bcast);

  }

  template<int Rank>
  void SumReduceLastDim(TensorAdapter<T>* ret) const {
    using namespace paddle::framework;
    auto in = EigenTensor<T, Rank>::From(_tensor);

    Eigen::array<int, 1> axis({ Rank - 1 });
    int newshape[Rank - 1];
    for (int i = 0; i < Rank - 1; ++i) {
        newshape[i] = _tensor.dims()[i];
    }
    auto out = EigenTensor<T, Rank - 1>::From(
        dynamic_cast<PaddleTensor<T>*>(ret)->_tensor, DDim(newshape, Rank - 1));

    auto &place = *eigen_device();
    out.device(place) = in.sum(axis);

  }

    void sum_reduce_last_dim(TensorAdapter<T>* ret) const {
        auto in_dims_size = shape().size();
        switch (in_dims_size) {
        case 1:
            SumReduceLastDim<1>(ret);
            break;
        case 2:
            SumReduceLastDim<2>(ret);
            break;
        case 3:
            SumReduceLastDim<3>(ret);
            break;
        case 4:
            SumReduceLastDim<4>(ret);
            break;
        case 5:
            SumReduceLastDim<5>(ret);
            break;
        default:
            throw std::invalid_argument("unsupported input dim size: " + std::to_string(in_dims_size));
        }
    }

  // slice by shape[0] of index
  // e.g. x.shape = [2, 3, 4]
  //      data of x[1] = x.slice(1, 2)
  //      x[1]->shape = [3, 4]
  std::shared_ptr<TensorAdapter<T>> operator[](size_t index);

  const std::shared_ptr<TensorAdapter<T>> operator[](size_t index) const;

private:
  paddle::platform::Place place() const { return _device_ctx->GetPlace(); }

  Eigen::DefaultDevice *eigen_device() const {
    // TODO: add other eigen device support
    return dynamic_cast<const paddle::platform::CPUDeviceContext *>(_device_ctx)
        ->eigen_device();
  }

private:
  paddle::framework::Tensor _tensor;

  const paddle::platform::DeviceContext *_device_ctx;

  size_t _scaling_factor;
};

class PaddleTensorFactory : public TensorAdapterFactory {
public:
  PaddleTensorFactory() = default;

  virtual ~PaddleTensorFactory() = default;

  std::shared_ptr<TensorAdapter<int64_t>>
  create_int64_t(const std::vector<size_t> &shape) override;

  std::shared_ptr<TensorAdapter<uint8_t>>
  create_uint8_t(const std::vector<size_t> &shape) override;

  std::shared_ptr<TensorAdapter<int64_t>> create_int64_t() override;

  PaddleTensorFactory(const paddle::platform::DeviceContext *device_ctx)
      : _device_ctx(device_ctx) {}

  const paddle::platform::DeviceContext *device_ctx() const {
    return _device_ctx;
  }

private:
  const paddle::platform::DeviceContext *_device_ctx;
};

} // namespace common

#include "paddle_tensor_impl.h"
