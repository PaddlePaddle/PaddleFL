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

#include <cmath>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace common {

using u128 = unsigned __int128;

template <typename T>
void PaddleTensor<T>::reshape(const std::vector<size_t> &shape) {
  std::vector<int64_t> shape_(shape.cbegin(), shape.cend());
  paddle::framework::DDim dim(shape_.data(), shape_.size());
  // 0 for default size
  _tensor.mutable_data<T>(dim, place(), 0);
}

template <typename T>
void PaddleTensor<T>::add(const TensorAdapter<T> *rhs,
                          TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  auto ret_ = dynamic_cast<PaddleTensor<T> *>(ret);

  PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                    "Input dims should be equal.");

  auto eigen_x = paddle::framework::EigenVector<T>::Flatten(_tensor);
  auto eigen_y = paddle::framework::EigenVector<T>::Flatten(rhs_->_tensor);
  auto eigen_z = paddle::framework::EigenVector<T>::Flatten(ret_->_tensor);

  auto &place = *eigen_device();

  eigen_z.device(place) = eigen_x + eigen_y;
}

template <typename T>
void PaddleTensor<T>::sub(const TensorAdapter<T> *rhs,
                          TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  auto ret_ = dynamic_cast<PaddleTensor<T> *>(ret);
  PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                    "Input dims should be equal.");

  auto eigen_x = paddle::framework::EigenVector<T>::Flatten(_tensor);
  auto eigen_y = paddle::framework::EigenVector<T>::Flatten(rhs_->_tensor);
  auto eigen_z = paddle::framework::EigenVector<T>::Flatten(ret_->_tensor);

  auto &place = *eigen_device();

  eigen_z.device(place) = eigen_x - eigen_y;
}

template <typename T>
void PaddleTensor<T>::mul(const TensorAdapter<T> *rhs,
                          TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  auto ret_ = dynamic_cast<PaddleTensor<T> *>(ret);
  PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                    "Input dims should be equal.");

  auto eigen_x = paddle::framework::EigenVector<T>::Flatten(_tensor);
  auto eigen_y = paddle::framework::EigenVector<T>::Flatten(rhs_->_tensor);
  auto eigen_z = paddle::framework::EigenVector<T>::Flatten(ret_->_tensor);

  auto &place = *eigen_device();

  eigen_z.device(place) = eigen_x * eigen_y;
}

template <typename T>
void PaddleTensor<T>::div(const TensorAdapter<T> *rhs,
                          TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                    "Input dims should be equal.");

  auto div_ = [](T a, T b) -> T { return a / b; };

  std::transform(data(), data() + numel(), rhs->data(), ret->data(), div_);
}

template <typename T>
void PaddleTensor<T>::mat_mul(const TensorAdapter<T> *rhs,
                              TensorAdapter<T> *ret,
                              bool transpose_lhs,
                              bool transpose_rhs) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  auto ret_ = dynamic_cast<PaddleTensor<T> *>(ret);

  auto &mat_a = _tensor;
  auto &mat_b = rhs_->_tensor;
  auto &mat_out = ret_->_tensor;

  // tensor with dims like [ h, w ] or [ batch_size , h, w ] is matrix
  auto is_matrix = [](const paddle::framework::Tensor &t) -> bool {
    return t.dims().size() == 2 || t.dims().size() == 3;
  };

  PADDLE_ENFORCE(mat_a.place() == mat_b.place() &&
                     mat_a.place() == mat_out.place(),
                 "The places of matrices must be same");

  PADDLE_ENFORCE(is_matrix(mat_a) && is_matrix(mat_b) && is_matrix(mat_out),
                 "The input and output of matmul must be matrix "
                 "or batched matrix.");

  PADDLE_ENFORCE(mat_a.dims().size() >= mat_b.dims().size(),
                 "Only following dims are supported: "
                 "Mat A is [BatchSize, H, W] and Mat B is [BatchSize, H, W]."
                 "Mat A is [BatchSize, H, W] and Mat B is [H, W]."
                 "Mat A is [H, W] and Mat B is [H, W].");

  using EigenTensor = paddle::framework::EigenTensor<T, 3>;
  using EigenTensor4 = paddle::framework::EigenTensor<T, 4>;
  using EigenTensor2 = paddle::framework::EigenTensor<T, 2>;

  auto to_const_eigen_tensor = [](const paddle::framework::Tensor &t) {
      auto dims = t.dims();
      if (dims.size() == 2) {
          dims = paddle::framework::make_ddim({1, dims[0], dims[1]});
      }
      return EigenTensor::From(t, dims);
  };

  auto to_eigen_tensor = [](paddle::framework::Tensor &t) {
      auto dims = t.dims();
      if (dims.size() == 2) {
          dims = paddle::framework::make_ddim({1, dims[0], 1, dims[1]});
      } else { // dims.size() == 3
          dims = paddle::framework::make_ddim({dims[0], dims[1], 1, dims[2]});
      }
      return EigenTensor4::From(t, dims);
  };


  auto &place = *eigen_device();

  auto t_a = to_const_eigen_tensor(mat_a);
  auto t_b = to_const_eigen_tensor(mat_b);
  auto t_c = to_eigen_tensor(mat_out);

  PADDLE_ENFORCE(t_a.dimension(2 - transpose_lhs) == t_b.dimension(1 + transpose_rhs),
                 "W_A != H_B.");

  auto batch_size = t_a.dimension(0);
  auto batch_size_b = t_b.dimension(0);

  PADDLE_ENFORCE(batch_size_b == batch_size || batch_size_b == 1,
                 "Mat B BatchSize mismatched.");

  PADDLE_ENFORCE(t_c.dimension(0) == batch_size,
                 "Result Mat BatchSize mismatched.");

  auto hc = t_c.dimension(1);
  auto wc = t_c.dimension(3);

  // matrix product of tensor contractions
  // please refer to
  // github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/CXX11/src/Tensor/README.md

    Eigen::array<Eigen::IndexPair<int>, 1> axis = {
        Eigen::IndexPair<int>(1 - transpose_lhs, 0 + transpose_rhs)};

#pragma omp for
    for (int i = 0; i < batch_size; ++i) {
        Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
            t_c_chip(t_c.data() + i * hc * wc, hc, wc);
        int idx_t_b = batch_size_b == 1 ? 0 : i;
        t_c_chip.device(place) = t_a.chip(i, 0).contract(t_b.chip(idx_t_b, 0), axis);
    }

}


template <typename T>
void PaddleTensor<T>::negative(TensorAdapter<T> *ret) const {

  auto neg_ = [](T a) -> T { return -a; };

  std::transform(data(), data() + numel(), ret->data(), neg_);
}

template <typename T>
void PaddleTensor<T>::bitwise_and(const TensorAdapter<T> *rhs,
                                  TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                    "Input dims should be equal.");

  auto and_ = [](T a, T b) -> T { return a & b; };

  std::transform(data(), data() + numel(), rhs->data(), ret->data(), and_);
}

template <typename T>
void PaddleTensor<T>::bitwise_or(const TensorAdapter<T> *rhs,
                                 TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                    "Input dims should be equal.");

  auto or_ = [](T a, T b) -> T { return a | b; };

  std::transform(data(), data() + numel(), rhs->data(), ret->data(), or_);
}

template <typename T>
void PaddleTensor<T>::bitwise_not(TensorAdapter<T> *ret) const {

  auto not_ = [](T a) -> T { return ~a; };

  std::transform(data(), data() + numel(), ret->data(), not_);
}

template <typename T>
void PaddleTensor<T>::bitwise_xor(const TensorAdapter<T> *rhs,
                                  TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                    "Input dims should be equal.");

  auto xor_ = [](T a, T b) -> T { return a ^ b; };

  std::transform(data(), data() + numel(), rhs->data(), ret->data(), xor_);
}

template <typename T>
void PaddleTensor<T>::lshift(size_t rhs, TensorAdapter<T> *ret) const {

  auto lshift_functor = [rhs](T a) -> T { return a << rhs; };

  std::transform(data(), data() + numel(), ret->data(), lshift_functor);
}

template <typename T>
void PaddleTensor<T>::rshift(size_t rhs, TensorAdapter<T> *ret) const {
  auto rshift_functor = [rhs](T a) -> T { return a >> rhs; };

  std::transform(data(), data() + numel(), ret->data(), rshift_functor);
}

template <typename T>
void PaddleTensor<T>::logical_rshift(size_t rhs, TensorAdapter<T> *ret) const {
  auto logical_rshift_functor = [rhs](T a) -> T {
    const size_t word_len = sizeof(T) * 8;
    T mask = (T)1 << word_len - rhs - 1;
    mask |= mask - 1;
    mask = rhs >= word_len ? 0 : mask;
    return a >> rhs & mask;
  };
  std::transform(data(), data() + numel(), ret->data(), logical_rshift_functor);
}

template <typename T>
void PaddleTensor<T>::add128(const TensorAdapter<T> *rhs,
                             TensorAdapter<T> *ret,
                             bool lhs_128,
                             bool rhs_128) const {
    PADDLE_ENFORCE_EQ(numel() / (1 + lhs_128),
                      rhs->numel() / (1 + rhs_128),
                      "Input numel should be equal.");

    using ConstType = Eigen::Tensor<const __int128, 1>;
    using Type = Eigen::Tensor<u128, 1>;

    size_t numel_ = ret->numel() / (sizeof(u128) / sizeof(T));

    Type x(numel_);
    for (size_t i = 0; i < numel_; ++i) {
        x(i) = lhs_128 ? *(reinterpret_cast<const u128*>(data()) + i) : *(data() + i);
    }

    Type y(numel_);
    for (size_t i = 0; i < numel_; ++i) {
        y(i) = rhs_128 ? *(reinterpret_cast<const u128*>(rhs->data()) +  i) : *(rhs->data() + i);
    }

    Eigen::TensorMap<Type> z(reinterpret_cast<u128*>(ret->data()), numel_);

    auto &place = *eigen_device();
    z.device(place) = x + y;
}

template <typename T>
void PaddleTensor<T>::sub128(const TensorAdapter<T> *rhs,
                             TensorAdapter<T> *ret,
                             bool lhs_128,
                             bool rhs_128) const {
    PADDLE_ENFORCE_EQ(numel() / (1 + lhs_128),
                      rhs->numel() / (1 + rhs_128),
                      "Input numel should be equal.");

    using ConstType = Eigen::Tensor<const u128, 1>;
    using Type = Eigen::Tensor<u128, 1>;

    size_t numel_ = ret->numel() / (sizeof(u128) / sizeof(T));

    Type x(numel_);
    for (size_t i = 0; i < numel_; ++i) {
        x(i) = lhs_128 ? *(reinterpret_cast<const u128*>(data()) +  i) : *(data() + i);
    }

    Type y(numel_);
    for (size_t i = 0; i < numel_; ++i) {
        y(i) = rhs_128 ? *(reinterpret_cast<const u128*>(rhs->data()) + i) : *(rhs->data() + i);
    }

    Eigen::TensorMap<Type> z(reinterpret_cast<u128*>(ret->data()), numel_);

    auto &place = *eigen_device();
    z.device(place) = x - y;
}

template <typename T>
void PaddleTensor<T>::mul128_with_truncate(const TensorAdapter<T> *rhs,
                             TensorAdapter<T> *ret,
                             bool lhs_128,
                             bool rhs_128) const {
    PADDLE_ENFORCE_EQ(numel() / (1 + lhs_128),
                      rhs->numel() / (1 + rhs_128),
                      "Input numel should be equal.");

    using ConstType = Eigen::Tensor<const u128, 1>;
    using Type = Eigen::Tensor<u128, 1>;

    size_t numel_ = ret->numel();

    Type x(numel_);
    for (size_t i = 0; i < numel_; ++i) {
        x(i) = lhs_128 ? *(reinterpret_cast<const u128*>(data()) + i) : *(data() + i);
    }

    Type y(numel_);
    for (size_t i = 0; i < numel_; ++i) {
        y(i) = rhs_128 ? *(reinterpret_cast<const u128*>(rhs->data()) + i) : *(rhs->data() + i);
    }

    Eigen::TensorMap<Eigen::Tensor<T, 1>> z(ret->data(), numel_);

    Type xy = x * y;

    Eigen::Tensor<T, 1> xy_trunc(numel_);

    // truncate
    for (size_t i = 0; i < numel_; ++i) {
        u128 tmp = xy(i);
        xy_trunc(i) = (T)(tmp >> _scaling_factor);
    }

    auto &place = *eigen_device();
    z.device(place) = xy_trunc;
}

template <typename T>
template <typename U>
PaddleTensor<T> &
PaddleTensor<T>::from_float_point_type(const paddle::framework::Tensor &tensor,
                                       size_t scaling_factor) {

  double scale = std::pow(2, scaling_factor);

  auto cast = [scale](U a) -> T { return a * scale; };

  _tensor.mutable_data<T>(tensor.dims(), place(), 0);

  std::transform(tensor.template data<U>(),
                 tensor.template data<U>() + tensor.numel(),
                 _tensor.template data<T>(), cast);

  this->scaling_factor() = scaling_factor;

  return *this;
}

template <typename T>
template <typename U>
PaddleTensor<T> &PaddleTensor<T>::from_float_point_scalar(
    const U &scalar, const std::vector<size_t> &shape, size_t scaling_factor) {

  double scale = std::pow(2, scaling_factor);

  auto trans = [scale, scalar](T) -> T { return scalar * scale; };

  reshape(shape);

  std::transform(_tensor.template data<T>(),
                 _tensor.template data<T>() + _tensor.numel(),
                 _tensor.template data<T>(), trans);

  this->scaling_factor() = scaling_factor;

  return *this;
}

template <typename T>
void PaddleTensor<T>::slice(size_t begin_idx, size_t end_idx,
                            TensorAdapter<T> *ret) const {
  auto ret_ = dynamic_cast<PaddleTensor<T> *>(ret);
  ret_->_tensor = _tensor.Slice(begin_idx, end_idx);

  ret->scaling_factor() = scaling_factor();
}

template<typename T>
std::shared_ptr<TensorAdapter<T>> PaddleTensor<T>::operator[](size_t index) {
    PADDLE_ENFORCE_GT(this->shape().size(), 1,
                     "lhs's shape must great than 1.");
    auto slice_shape = this->shape();
    slice_shape.erase(slice_shape.begin());
    std::shared_ptr<PaddleTensor<T>> ret = std::make_shared<PaddleTensor<T>>(_device_ctx);
    ret->reshape(slice_shape);

    this->slice(index, index + 1, ret.get());
    ret->reshape(slice_shape);
    return ret;
}

template<typename T>
const std::shared_ptr<TensorAdapter<T>> PaddleTensor<T>::operator[](size_t index) const {
    return const_cast<PaddleTensor*>(this)->operator[](index);
}

} // namespace common
