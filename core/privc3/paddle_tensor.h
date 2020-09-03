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

#include "core/paddlefl_mpc/operators/math/math_function.h"

#include "tensor_adapter.h"
#include "tensor_adapter_factory.h"

namespace aby3 {

template <typename T> class PaddleTensor : public TensorAdapter<T> {
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

  paddle::framework::Tensor* paddle_tensor() {
    return &_tensor;
  }

  std::vector<size_t> shape() const override {
    return paddle::framework::vectorize<size_t>(_tensor.dims());
  }

  size_t numel() const override { return _tensor.numel(); }

  void add(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void sub(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void negative(TensorAdapter<T> *ret) const override;

  void mul(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void div(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override;

  void mat_mul(const TensorAdapter<T> *rhs,
               TensorAdapter<T> *ret) const override;

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
  void Transpose(const std::vector<int> axis, TensorAdapter<T>* ret) {
    paddle::operators::math::Transpose<paddle::platform::CPUDeviceContext, T, Rank> trans;
    trans(*(dynamic_cast<const paddle::platform::CPUDeviceContext*>(_device_ctx)),
          _tensor,
          dynamic_cast<PaddleTensor<T>*>(ret)->paddle_tensor(),
          axis);
  }

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

  std::shared_ptr<TensorAdapter<int64_t>> create_int64_t() override;

  PaddleTensorFactory(const paddle::platform::DeviceContext *device_ctx)
      : _device_ctx(device_ctx) {}

  const paddle::platform::DeviceContext *device_ctx() const {
    return _device_ctx;
  }

private:
  const paddle::platform::DeviceContext *_device_ctx;
};

} // namespace aby3

#include "paddle_tensor_impl.h"
