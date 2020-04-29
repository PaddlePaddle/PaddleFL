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

namespace aby3 {

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
                              TensorAdapter<T> *ret) const {
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  auto ret_ = dynamic_cast<PaddleTensor<T> *>(ret);

  auto &mat_a = _tensor;
  auto &mat_b = rhs_->_tensor;
  auto &mat_out = ret_->_tensor;

  // tensor with dims like [ m, n ] or [ 1, m, n ] is matrix
  auto is_matrix = [](const paddle::framework::Tensor &t) -> bool {
    return t.dims().size() == 2 || t.dims().size() == 3 && t.dims()[0] == 1;
  };

  PADDLE_ENFORCE(is_matrix(mat_a) && is_matrix(mat_b) && is_matrix(mat_out),
                 "The input and output of matmul must be matrix.");

  PADDLE_ENFORCE(mat_a.place() == mat_b.place() &&
                     mat_a.place() == mat_out.place(),
                 "The places of matrices must be same");

  auto dim_a = mat_a.dims();
  auto dim_b = mat_b.dims();
  auto dim_out = mat_out.dims();

  PADDLE_ENFORCE_EQ(dim_a[dim_a.size() - 1], dim_b[dim_b.size() - 2]);

  using EigenMatrix = paddle::framework::EigenMatrix<T>;

  // matrix product of tensor contractions
  // please refer to
  // github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/CXX11/src/Tensor/README.md
  // flatten tensor to 2d, 1 for num_col_dims, assum input dims are 2
  typename EigenMatrix::ConstType mat_lhs =
      EigenMatrix::Reshape(mat_a, dim_a.size() - 1);
  typename EigenMatrix::ConstType mat_rhs =
      EigenMatrix::Reshape(mat_b, dim_b.size() - 1);
  typename EigenMatrix::Type mat_ret =
      EigenMatrix::Reshape(mat_out, dim_out.size() - 1);

  auto &place = *eigen_device();

  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(1, 0)};

  mat_ret.device(place) = mat_lhs.contract(mat_rhs, product_dims);
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

} // namespace aby3
