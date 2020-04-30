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

#include <algorithm>
#include <memory>

#include "paddle/fluid/platform/enforce.h"
#include "prng.h"

namespace aby3 {

template <typename T, size_t N>
FixedPointTensor<T, N>::FixedPointTensor(TensorAdapter<T> *share_tensor[2]) {
  // TODO: check tensors' shapes
  _share[0] = share_tensor[0];
  _share[1] = share_tensor[1];
}

template <typename T, size_t N>
FixedPointTensor<T, N>::FixedPointTensor(TensorAdapter<T> *share_tensor_0,
                                         TensorAdapter<T> *share_tensor_1) {
  // TODO: check tensors' shapes
  _share[0] = share_tensor_0;
  _share[1] = share_tensor_1;
}

template <typename T, size_t N>
TensorAdapter<T> *FixedPointTensor<T, N>::mutable_share(size_t idx) {
  PADDLE_ENFORCE_LT(idx, 2, "Input should be less than 2.");
  return _share[idx];
}

template <typename T, size_t N>
const TensorAdapter<T> *FixedPointTensor<T, N>::share(size_t idx) const {
  PADDLE_ENFORCE_LT(idx, 2, "Input should be less than 2.");
  return _share[idx];
}

// reveal fixedpointtensor to one party
template <typename T, size_t N>
void FixedPointTensor<T, N>::reveal_to_one(size_t party,
                                           TensorAdapter<T> *ret) const {

  if (party == this->party()) {
    // TODO: check if tensor shape equal

    auto buffer = tensor_factory()->template create<T>(ret->shape());
    aby3_ctx()->network()->template recv(pre_party(), *buffer);

    share(0)->add(buffer.get(), ret);
    share(1)->add(ret, ret);
    ret->scaling_factor() = N;

  } else if (party == next_party()) {

    aby3_ctx()->network()->template send(party, *share(0));
  }
}

// reveal fixedpointtensor to all parties
template <typename T, size_t N>
void FixedPointTensor<T, N>::reveal(TensorAdapter<T> *ret) const {
  for (size_t i = 0; i < 3; ++i) {
    reveal_to_one(i, ret);
  }
}

template <typename T, size_t N>
const std::vector<size_t> FixedPointTensor<T, N>::shape() const {
  return _share[0]->shape();
}

// convert TensorAdapter to shares
template <typename T, size_t N>
void FixedPointTensor<T, N>::share(const TensorAdapter<T> *input,
                                   TensorAdapter<T> *output_shares[3],
                                   block seed) {

  if (equals(seed, g_zero_block)) {
    seed = block_from_dev_urandom();
  }
  // set seed of prng[2]
  aby3_ctx()->set_random_seed(seed, 2);

  aby3_ctx()->template gen_random_private(*output_shares[0]);
  aby3_ctx()->template gen_random_private(*output_shares[1]);

  auto temp = tensor_factory()->template create<T>(input->shape());
  output_shares[0]->add(output_shares[1], temp.get());
  input->sub(temp.get(), output_shares[2]);
  for (int i = 0; i < 3; ++i) {
    output_shares[i]->scaling_factor() = input->scaling_factor();
  }
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::add(const FixedPointTensor<T, N> *rhs,
                                 FixedPointTensor<T, N> *ret) const {
  _share[0]->add(rhs->_share[0], ret->_share[0]);
  _share[1]->add(rhs->_share[1], ret->_share[1]);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::add(const TensorAdapter<T> *rhs,
                                 FixedPointTensor<T, N> *ret) const {
  PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(), "no match scaling factor");
  if (party() == 0) {
    _share[0]->add(rhs, ret->_share[0]);
    _share[1]->copy(ret->_share[1]);
  } else if (party() == 1) {
    _share[0]->copy(ret->_share[0]);
    _share[1]->copy(ret->_share[1]);
  } else {
    _share[0]->copy(ret->_share[0]);
    _share[1]->add(rhs, ret->_share[1]);
  }
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::sub(const FixedPointTensor<T, N> *rhs,
                                 FixedPointTensor<T, N> *ret) const {
  _share[0]->sub(rhs->_share[0], ret->_share[0]);
  _share[1]->sub(rhs->_share[1], ret->_share[1]);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::sub(const TensorAdapter<T> *rhs,
                                 FixedPointTensor<T, N> *ret) const {
  PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(), "no match scaling factor");
  if (party() == 0) {
    _share[0]->sub(rhs, ret->_share[0]);
    _share[1]->copy(ret->_share[1]);
  } else if (party() == 1) {
    _share[0]->copy(ret->_share[0]);
    _share[1]->copy(ret->_share[1]);
  } else {
    _share[0]->copy(ret->_share[0]);
    _share[1]->sub(rhs, ret->_share[1]);
  }
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::negative(FixedPointTensor<T, N> *ret) const {
  _share[0]->negative(ret->_share[0]);
  _share[1]->negative(ret->_share[1]);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::mul(const FixedPointTensor<T, N> *rhs,
                                 FixedPointTensor<T, N> *ret) const {

  auto r_zero = tensor_factory()->template create<T>(this->shape());
  aby3_ctx()->gen_zero_sharing_arithmetic(*r_zero.get());

  // temp = _share[0] * rhs->_share[0] +
  //        _share[0] * rhs->_share[1] +
  //        _share[1] * rhs->_share[0] +
  //        r_zero
  auto temp = tensor_factory()->template create<T>(this->shape());
  auto temp1 = tensor_factory()->template create<T>(this->shape());

  _share[0]->mul(rhs->_share[0], temp.get());
  _share[0]->mul(rhs->_share[1], temp1.get());
  temp1->add(temp.get(), temp1.get());

  _share[1]->mul(rhs->_share[0], temp.get());
  temp1->add(r_zero.get(), temp1.get());
  temp->add(temp1.get(), temp.get());

  auto temp2 = tensor_factory()->template create<T>(this->shape());
  auto temp3 = tensor_factory()->template create<T>(this->shape());

  TensorAdapter<int64_t> *temp_array[2] = {temp2.get(), temp3.get()};

  std::shared_ptr<FixedPointTensor<T, N>> ret_no_trunc =
      std::make_shared<FixedPointTensor<T, N>>(temp_array);
  temp->copy(ret_no_trunc->_share[0]);
  reshare(temp.get(), ret_no_trunc->_share[1]);

  truncate1(ret_no_trunc.get(), ret, N);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::truncate1(FixedPointTensor<T, N> *op,
                                       FixedPointTensor<T, N> *ret,
                                       size_t scaling_factor) {
  // implement ABY3's truncate1 algorithm
  if (party() == 0) {
    // party0
    op->_share[0]->rshift(scaling_factor, ret->_share[0]);
    aby3_ctx()->network()->template recv(1, *(ret->_share[1]));

  } else if (party() == 1) {
    // party1
    auto r_12 = tensor_factory()->template create<T>(op->shape());
    aby3_ctx()->template gen_random(*r_12.get(), true);

    op->_share[0]->add(op->_share[1], ret->_share[0]);
    ret->_share[0]->rshift(scaling_factor, ret->_share[0]);
    ret->_share[0]->sub(r_12.get(), ret->_share[0]);

    aby3_ctx()->network()->template send(0, *(ret->_share[0]));
    r_12->copy(ret->_share[1]);

  } else {
    // party2
    op->_share[1]->rshift(scaling_factor, ret->_share[1]);

    auto r_21 = tensor_factory()->template create<T>(op->shape());
    aby3_ctx()->template gen_random(*r_21.get(), false);

    r_21->copy(ret->_share[0]);
  }
  return;
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::mul2(const FixedPointTensor<T, N> *rhs,
                                  FixedPointTensor<T, N> *ret) const {

  // element-wise mul implemented by ABY3's truncate2 algorithm
  std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
  for (int i = 0; i < 12; ++i) {
    temp.emplace_back(tensor_factory()->template create<T>(this->shape()));
  }

  // gen boolean random share
  aby3_ctx()->template gen_random(*temp[0], 0);
  aby3_ctx()->template gen_random(*temp[1], 1);

  std::shared_ptr<BooleanTensor<T>> r =
      std::make_shared<BooleanTensor<T>>(temp[0].get(), temp[1].get());
  std::shared_ptr<BooleanTensor<T>> r_integer =
      std::make_shared<BooleanTensor<T>>(temp[4].get(), temp[5].get());
  r->rshift(N, r_integer.get());

  std::shared_ptr<FixedPointTensor<T, N>> r_fixed =
      std::make_shared<FixedPointTensor<T, N>>(temp[6].get(), temp[7].get());
  std::shared_ptr<FixedPointTensor<T, N>> r_integer_fixed =
      std::make_shared<FixedPointTensor<T, N>>(temp[8].get(), temp[9].get());
  r->b2a(r_fixed.get());

  // r'
  r_integer->b2a(r_integer_fixed.get());

  // r_zero = gen_zero_share(_shape[0]->shape)
  auto r_zero = tensor_factory()->template create<T>(this->shape());
  aby3_ctx()->template gen_zero_sharing_arithmetic(*r_zero);

  // temp[10] = _share[0] * rhs->_share[0] +
  //        _share[0] * rhs->_share[1] +
  //        _share[1] * rhs->_share[0] +
  //        r_zero - r[0]
  _share[0]->mul(rhs->_share[0], temp[11].get());
  _share[0]->mul(rhs->_share[1], temp[10].get());
  temp[11]->add(temp[10].get(), temp[11].get());
  _share[1]->mul(rhs->_share[0], temp[10].get());
  temp[11]->add(temp[10].get(), temp[11].get());
  r_zero->sub(r_fixed->_share[0], temp[10].get());
  temp[10]->add(temp[11].get(), temp[10].get());

  // ret = reshare
  temp[10]->copy(ret->_share[0]);
  reshare(temp[10].get(), ret->_share[1]);

  // ret = reconstruct(ret).rshift(N)
  // ret = ret + r'
  ret->reveal(temp[10].get());
  temp[10]->rshift(N, temp[10].get());
  r_integer_fixed->add(temp[10].get(), ret);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::mul(const TensorAdapter<T> *rhs,
                                 FixedPointTensor<T, N> *ret) const {
  // PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
  //                   "no match scaling factor");
  auto temp0 = tensor_factory()->template create<T>(this->shape());
  auto temp1 = tensor_factory()->template create<T>(this->shape());
  std::shared_ptr<FixedPointTensor<T, N>> temp =
      std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());

  _share[0]->mul(rhs, temp->_share[0]);
  _share[1]->mul(rhs, temp->_share[1]);
  truncate1(temp.get(), ret, rhs->scaling_factor());
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::sum(FixedPointTensor<T, N> *ret) const {
  PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");
  T sum1 = (T)0;
  T sum2 = (T)0;
  T *iter_0 = _share[0]->data();
  T *iter_1 = _share[1]->data();
  for (int i = 0; i < this->numel(); ++i) {
    sum1 += *(iter_0 + i);
    sum2 += *(iter_1 + i);
  }
  assign_to_tensor(ret->_share[0], sum1);
  assign_to_tensor(ret->_share[1], sum2);
}

template <typename T, size_t N>
template <template <typename U, size_t...> class CTensor, size_t... N1>
void FixedPointTensor<T, N>::dot_mul(const CTensor<T, N1...> *rhs,
                                     FixedPointTensor<T, N> *ret) const {
  PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");

  auto temp0 = tensor_factory()->template create<T>(this->shape());
  auto temp1 = tensor_factory()->template create<T>(this->shape());
  std::shared_ptr<FixedPointTensor<T, N>> temp =
      std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
  this->mul(rhs, temp.get());
  temp->sum(ret);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::mat_mul(const FixedPointTensor<T, N> *rhs,
                                     FixedPointTensor<T, N> *ret) const {

  auto r_zero = tensor_factory()->template create<T>(ret->shape());
  aby3_ctx()->gen_zero_sharing_arithmetic(*r_zero.get());

  // temp = _share[0]->mat_mul(rhs->_share[0]) +
  //        _share[0]->mat_mul(rhs->_share[1]) +
  //        _share[1]->mat_mul(rhs->_share[0]) +
  //        r_zero
  auto temp = tensor_factory()->template create<T>(ret->shape());
  auto temp1 = tensor_factory()->template create<T>(ret->shape());

  _share[0]->mat_mul(rhs->_share[0], temp.get());
  _share[0]->mat_mul(rhs->_share[1], temp1.get());
  temp1->add(temp.get(), temp1.get());

  _share[1]->mat_mul(rhs->_share[0], temp.get());
  temp1->add(r_zero.get(), temp1.get());
  temp->add(temp1.get(), temp.get());

  auto temp2 = tensor_factory()->template create<T>(ret->shape());
  auto temp3 = tensor_factory()->template create<T>(ret->shape());

  TensorAdapter<int64_t> *temp_array[2] = {temp2.get(), temp3.get()};

  std::shared_ptr<FixedPointTensor<T, N>> ret_no_trunc =
      std::make_shared<FixedPointTensor<T, N>>(temp_array);

  temp->copy(ret_no_trunc->_share[0]);
  reshare(temp.get(), ret_no_trunc->_share[1]);

  truncate1(ret_no_trunc.get(), ret, N);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::mat_mul(const TensorAdapter<T> *rhs,
                                     FixedPointTensor<T, N> *ret) const {
  _share[0]->mat_mul(rhs, ret->_share[0]);
  _share[1]->mat_mul(rhs, ret->_share[1]);
  truncate1(ret, ret, rhs->scaling_factor());
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::div(const TensorAdapter<T> *rhs,
                                 FixedPointTensor<T, N> *ret) const {
  PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(), "no match scaling factor");

  auto temp = tensor_factory()->template create<T>(this->shape());

  double scale = std::pow(2, rhs->scaling_factor());
  auto inverse = [scale](T d) -> T { return 1.0 * scale / d * scale; };
  std::transform(rhs->data(), rhs->data() + rhs->numel(), temp->data(),
                 inverse);
  temp->scaling_factor() = rhs->scaling_factor();

  this->mul(temp.get(), ret);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::exp(FixedPointTensor<T, N> *ret,
                                 size_t iter) const {
  // exp approximate: exp(x) = \lim_{n->inf} (1+x/n)^n
  auto pow_iter = tensor_factory()->template create<T>(this->shape());
  assign_to_tensor(pow_iter.get(), (T)(pow(2, N - iter)));
  pow_iter->scaling_factor() = N;

  auto tensor_one = tensor_factory()->template create<T>(this->shape());
  assign_to_tensor(tensor_one.get(), (T)1 << N);
  tensor_one->scaling_factor() = N;

  this->mul(pow_iter.get(), ret);

  ret->add(tensor_one.get(), ret);

  for (int i = 0; i < iter; ++i) {
    ret->mul(ret, ret);
  }
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::relu(FixedPointTensor<T, N> *ret) const {
  // utilize polynomial_piecewise
  // break_point = {0}, coeff[0] = {0, 0}, coeff[1] = {0, 1}
  // break_point.shape = {1, this->shape}, coeff.shape = {2, 2, this->shape}

  auto shape_ = shape();
  // construct break_point
  auto b_shape = shape_;
  b_shape.insert(b_shape.begin(), 1);

  auto break_point = tensor_factory()->template create<T>(b_shape);

  T *b_ptr = break_point->data();
  for (size_t i = 0; i < break_point->numel(); ++i) {
    b_ptr[i] = 0;
  }
  break_point->scaling_factor() = N;

  // contruct coeff
  std::vector<size_t> c_shape = {2, 2};
  c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());

  auto coeff = tensor_factory()->template create<T>(c_shape);

  T *c_ptr = coeff->data();

  for (size_t i = 0; i < 3 * this->numel(); ++i) {
    c_ptr[i] = 0;
  }
  for (size_t i = 3 * this->numel(); i < 4 * this->numel(); ++i) {
    c_ptr[i] = (T)1 << N;
  }
  coeff->scaling_factor() = N;

  this->polynomial_piecewise(coeff.get(), break_point.get(), ret);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid(FixedPointTensor<T, N> *ret) const {
  // utilize polynomial_piecewise
  // break_point = {-2.5, 2.5}
  // coeff[0] = {10^-4, 0}, coeff[1] = {0.5, 0.17}
  // coeff[2] = {1 - 10^-4, 0}
  // break_point.shape = {2, this->shape}, coeff.shape = {3, 2, this->shape}

  // construct break_point
  auto shape_ = shape();
  // construct break_point
  auto b_shape = shape_;
  b_shape.insert(b_shape.begin(), 2);

  auto break_point = tensor_factory()->template create<T>(b_shape);

  T *b_ptr = break_point->data();
  for (size_t i = 0; i < break_point->numel(); ++i) {
    b_ptr[i] = 0;
  }
  for (size_t i = 0; i < break_point->numel() / 2; ++i) {
    b_ptr[i] = (T)(-2.5 * pow(2, N));
  }
  for (size_t i = break_point->numel() / 2; i < break_point->numel(); ++i) {
    b_ptr[i] = (T)(2.5 * pow(2, N));
  }
  break_point->scaling_factor() = N;

  // contruct coeff
  std::vector<size_t> c_shape = {3, 2};
  c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());

  auto coeff = tensor_factory()->template create<T>(c_shape);

  T *c_ptr = coeff->data();

  size_t numel = this->numel();
  double scale = std::pow(2, N);
  for (size_t i = 0; i < numel; ++i) {
    c_ptr[i] = 0.0001 * scale;
    c_ptr[i + numel] = 0;
    c_ptr[i + 2 * numel] = 0.5 * scale;
    c_ptr[i + 3 * numel] = 0.17 * scale;
    c_ptr[i + 4 * numel] = (1 - 0.0001) * scale;
    c_ptr[i + 5 * numel] = 0;
  }
  coeff->scaling_factor() = N;

  this->polynomial_piecewise(coeff.get(), break_point.get(), ret);
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::softmax(FixedPointTensor<T, N> *ret) const {
  // relu_x = relu(this)
  auto &shape = this->shape();
  auto temp0 = tensor_factory()->template create<T>(this->shape());
  auto temp1 = tensor_factory()->template create<T>(this->shape());
  std::shared_ptr<FixedPointTensor<T, N>> relu_x =
      std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
  this->relu(relu_x.get());

  // get sum: reduce shape : from this->shape() to
  // this->shape()[0],...,shape()[n-2]
  std::vector<size_t> shape_sum;
  for (int i = 0; i < shape.size() - 1; ++i) {
    shape_sum.emplace_back(shape[i]);
  }

  auto temp2 = tensor_factory()->template create<T>(shape_sum);
  auto temp3 = tensor_factory()->template create<T>(shape_sum);
  std::shared_ptr<FixedPointTensor<T, N>> sum =
      std::make_shared<FixedPointTensor<T, N>>(temp2.get(), temp3.get());

  // reduce relu_x's last dim
  reduce(relu_x.get(), sum.get());

  // reveal (TODO: security improve)
  auto sum_plain = tensor_factory()->template create<T>(sum->shape());
  sum->reveal(sum_plain.get());

  // extend sum_plain shape to relu_x->shape(), padding with sum_value
  auto sum_extend = tensor_factory()->template create<T>(relu_x->shape());
  sum_extend->scaling_factor() = N;
  T *sum_ext_ptr = sum_extend->data();
  T *sum_plain_ptr = sum_plain->data();

  size_t ite_size = shape[shape.size() - 1];
  for (int j = 0; j < sum_plain->numel(); ++j) {
    for (int i = 0; i < ite_size; ++i) {
      *(sum_ext_ptr + j * ite_size + i) = *(sum_plain_ptr + j);
    }
  }

  relu_x->div(sum_extend.get(), ret);
}

// reduce last dim
template <typename T, size_t N>
void FixedPointTensor<T, N>::reduce(FixedPointTensor<T, N> *input,
                                    FixedPointTensor<T, N> *ret) {
  // enfoce shape: input->shape[0 ... (n-2)] == ret shape
  auto &shape = input->shape();
  size_t ite_size = shape[shape.size() - 1];

  T *ret_begin_ptr_0 = ret->_share[0]->data();
  T *ret_begin_ptr_1 = ret->_share[1]->data();

  T *input_begin_ptr_0 = input->_share[0]->data();
  T *input_begin_ptr_1 = input->_share[1]->data();

  for (int j = 0; j < ret->numel(); ++j) {
    *(ret_begin_ptr_0 + j) = *(input_begin_ptr_0 + j * ite_size);
    *(ret_begin_ptr_1 + j) = *(input_begin_ptr_1 + j * ite_size);
    for (int i = 1; i < ite_size; ++i) {
      *(ret_begin_ptr_0 + j) += *(input_begin_ptr_0 + j * ite_size + i);
      *(ret_begin_ptr_1 + j) += *(input_begin_ptr_1 + j * ite_size + i);
    }
  }
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::polynomial(const TensorAdapter<T> *coeff,
                                        FixedPointTensor<T, N> *ret) const {

  // e.g., x.shape = {2, 3}, coeff.shape = {n, 2, 3} (n: polynomial power)

  // TODO: improve performance: [ABY3]
  std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
  for (int i = 0; i < 5; ++i) {
    temp.emplace_back(tensor_factory()->template create<T>(this->shape()));
  }
  std::shared_ptr<FixedPointTensor<T, N>> x_pow_i =
      std::make_shared<FixedPointTensor<T, N>>(temp[0].get(), temp[1].get());
  std::shared_ptr<FixedPointTensor<T, N>> temp_fixed =
      std::make_shared<FixedPointTensor<T, N>>(temp[2].get(), temp[3].get());

  assign_to_tensor(ret->_share[0], (T)0);
  assign_to_tensor(ret->_share[1], (T)0);

  // x_pow_i.get() = 1;
  assign_to_tensor(x_pow_i.get()->_share[0], (T)0);
  assign_to_tensor(x_pow_i.get()->_share[1], (T)0);
  assign_to_tensor(temp[4].get(), (T)1 << N);
  temp[4]->scaling_factor() = N;
  x_pow_i->add(temp[4].get(), x_pow_i.get());

  for (int i = 0; i < coeff->shape()[0]; ++i) {
    auto t = tensor_factory()->template create<T>();
    coeff->slice(i, i + 1, t.get());
    auto t_shape = t->shape();
    // remove leading 1
    t_shape.erase(t_shape.begin());
    t->reshape(t_shape);
    x_pow_i->mul(t.get(), temp_fixed.get());
    ret->add(temp_fixed.get(), ret);
    x_pow_i->mul(this, x_pow_i.get());
  }
}

template <typename T, size_t N>
void FixedPointTensor<T, N>::polynomial_piecewise(
    const TensorAdapter<T> *coeff, const TensorAdapter<T> *break_point,
    FixedPointTensor<T, N> *ret) const {

  // e.g., x.shape = {2, 3},
  // break_point.shape = {k, 2, 3} (k: num of break point)
  //       coeff.shape = {k + 1, n, 2, 3} (n: poly power)

  std::vector<std::shared_ptr<BooleanTensor<T>>> msb;

  int len_break_point = break_point->shape()[0];
  int len_coeff = coeff->shape()[0];

  // number of temp tensor used
  int temp_total =
      4 * len_break_point + 2 + 2 * (len_break_point - 1) + 2 + 4 * len_coeff;
  std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
  for (int i = 0; i < temp_total; ++i) {
    temp.emplace_back(tensor_factory()->template create<T>(this->shape()));
  }
  int temp_index = 0;

  // std::vector<std::shared_ptr<TensorAdapter<T>>> paddle_t_break;
  std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp1;

  for (int i = 0; i < break_point->shape()[0]; ++i) {
    // msb[i] = msb(x - break_point[i])
    auto t_break = tensor_factory()->template create<T>();
    break_point->slice(i, i + 1, t_break.get());

    auto t_shape = t_break->shape();
    // remove leading 1
    t_shape.erase(t_shape.begin());
    t_break->reshape(t_shape);

    temp1.emplace_back(std::make_shared<FixedPointTensor<T, N>>(
        temp[temp_index++].get(), temp[temp_index++].get()));
    this->sub(t_break.get(), temp1[i].get());
    msb.emplace_back(std::make_shared<BooleanTensor<T>>(
        temp[temp_index++].get(), temp[temp_index++].get()));
    msb[i]->bit_extract(sizeof(T) * 8 - 1, temp1[i].get());
  }

  // b[0] = msb[0], b[i + 1] = ~ msb[i] & msb[i + 1]
  std::vector<std::shared_ptr<BooleanTensor<T>>> b;
  b.emplace_back(std::make_shared<BooleanTensor<T>>(temp[temp_index++].get(),
                                                    temp[temp_index++].get()));
  b[0] = msb[0];

  for (int i = 0; i < len_break_point - 1; ++i) {
    b.emplace_back(std::make_shared<BooleanTensor<T>>(
        temp[temp_index++].get(), temp[temp_index++].get()));

    msb[i]->bitwise_not(b[i + 1].get());
    b[i + 1]->bitwise_and(msb[i + 1].get(), b[i + 1].get());
  }

  b.emplace_back(std::make_shared<BooleanTensor<T>>(temp[temp_index++].get(),
                                                    temp[temp_index++].get()));
  msb[len_break_point - 1]->bitwise_not(b[len_break_point].get());

  // ret += b[i].mul(polynomial(coeff[i]))
  std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp_fixed;
  std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp_fixed1;

  assign_to_tensor(ret->_share[0], (T)0);
  assign_to_tensor(ret->_share[1], (T)0);

  for (int i = 0; i < len_coeff; ++i) {
    temp_fixed.emplace_back(std::make_shared<FixedPointTensor<T, N>>(
        temp[temp_index++].get(), temp[temp_index++].get()));
    temp_fixed1.emplace_back(std::make_shared<FixedPointTensor<T, N>>(
        temp[temp_index++].get(), temp[temp_index++].get()));
    auto t = tensor_factory()->template create<T>();
    coeff->slice(i, i + 1, t.get());
    auto t_shape = t->shape();
    // remove leading 1
    t_shape.erase(t_shape.begin());
    t->reshape(t_shape);
    ;
    this->polynomial(t.get(), temp_fixed[i].get());
    b[i]->bit_extract(0, b[i].get());
    b[i]->mul(temp_fixed[i].get(), temp_fixed1[i].get());
    ret->add(temp_fixed1[i].get(), ret);
  }
}

template <typename T, size_t N>
template <template <typename U, size_t...> class CTensor, size_t... N1>
void FixedPointTensor<T, N>::lt(const CTensor<T, N1...> *rhs,
                                BooleanTensor<T> *ret) const {

  std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
  for (int i = 0; i < 2; ++i) {
    temp.emplace_back(tensor_factory()->template create<T>(this->shape()));
  }
  std::shared_ptr<FixedPointTensor<T, N>> sub_result =
      std::make_shared<FixedPointTensor<T, N>>(temp[0].get(), temp[1].get());
  this->sub(rhs, sub_result.get());
  ret->bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template <typename T, size_t N>
template <template <typename U, size_t...> class CTensor, size_t... N1>
void FixedPointTensor<T, N>::leq(const CTensor<T, N1...> *rhs,
                                 BooleanTensor<T> *ret) const {

  this->gt(rhs, ret);
  auto tensor_one = tensor_factory()->template create<T>(this->shape());

  assign_to_tensor(tensor_one.get(), (T)1);
  ret->bitwise_xor(tensor_one.get(), ret);
}

template <typename T, size_t N>
template <template <typename U, size_t...> class CTensor, size_t... N1>
void FixedPointTensor<T, N>::gt(const CTensor<T, N1...> *rhs,
                                BooleanTensor<T> *ret) const {

  std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
  for (int i = 0; i < 2; ++i) {
    temp.emplace_back(tensor_factory()->template create<T>(this->shape()));
  }
  std::shared_ptr<FixedPointTensor<T, N>> sub_result =
      std::make_shared<FixedPointTensor<T, N>>(temp[0].get(), temp[1].get());
  this->sub(rhs, sub_result.get());
  sub_result->negative(sub_result.get());
  ret->template bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template <typename T, size_t N>
template <template <typename U, size_t...> class CTensor, size_t... N1>
void FixedPointTensor<T, N>::geq(const CTensor<T, N1...> *rhs,
                                 BooleanTensor<T> *ret) const {

  this->lt(rhs, ret);
  auto tensor_one = tensor_factory()->template create<T>(this->shape());

  assign_to_tensor(tensor_one.get(), (T)1);
  ret->bitwise_xor(tensor_one.get(), ret);
}

template <typename T, size_t N>
template <template <typename U, size_t...> class CTensor, size_t... N1>
void FixedPointTensor<T, N>::eq(const CTensor<T, N1...> *rhs,
                                BooleanTensor<T> *ret) const {

  this->neq(rhs, ret);
  auto tensor_one = tensor_factory()->template create<T>(this->shape());
  assign_to_tensor(tensor_one.get(), (T)1);
  ret->bitwise_xor(tensor_one.get(), ret);
}

template <typename T, size_t N>
template <template <typename U, size_t...> class CTensor, size_t... N1>
void FixedPointTensor<T, N>::neq(const CTensor<T, N1...> *rhs,
                                 BooleanTensor<T> *ret) const {
  std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(tensor_factory()->template create<T>(this->shape()));
  }
  std::shared_ptr<BooleanTensor<T>> lt =
      std::make_shared<BooleanTensor<T>>(temp[0].get(), temp[1].get());
  std::shared_ptr<BooleanTensor<T>> gt =
      std::make_shared<BooleanTensor<T>>(temp[2].get(), temp[3].get());

  this->lt(rhs, lt.get());
  this->gt(rhs, gt.get());
  lt->bitwise_or(gt.get(), ret);
}

template <typename T>
inline void assign_to_tensor(TensorAdapter<T> *input, T assign_num) {
  size_t size_one_dim = input->numel();
  T *iter = input->data();
  for (int i = 0; i < size_one_dim; ++i) {
    *(iter + i) = assign_num;
  }
}

} // namespace aby3
