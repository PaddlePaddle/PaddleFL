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

#include "boolean_tensor.h"
#include "circuit_context.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "paddle_tensor.h"

namespace aby3 {

template <typename T, size_t N> class FixedPointTensor {

public:
  explicit FixedPointTensor(TensorAdapter<T> *share_tensor[2]);

  explicit FixedPointTensor(TensorAdapter<T> *share_tensor_0,
                            TensorAdapter<T> *share_tensor_1);

  ~FixedPointTensor(){};

  // get mutable shape of tensor
  TensorAdapter<T> *mutable_share(size_t idx);

  const TensorAdapter<T> *share(size_t idx) const;

  size_t numel() const { return _share[0]->numel(); }

  // reveal fixedpointtensor to one party
  void reveal_to_one(size_t party, TensorAdapter<T> *ret) const;

  // reveal fixedpointtensor to all parties
  void reveal(TensorAdapter<T> *ret) const;

  const std::vector<size_t> shape() const;

  // convert TensorAdapter to shares
  static void share(const TensorAdapter<T> *input,
                    TensorAdapter<T> *output_shares[3],
                    block seed = g_zero_block);

  // element-wise add with FixedPointTensor
  void add(const FixedPointTensor *rhs, FixedPointTensor *ret) const;

  // element-wise add with TensorAdapter

  void add(const TensorAdapter<T> *rhs, FixedPointTensor *ret) const;

  // element-wise sub with FixedPointTensor
  void sub(const FixedPointTensor *rhs, FixedPointTensor *ret) const;

  // element-wise sub with TensorAdapter
  void sub(const TensorAdapter<T> *rhs, FixedPointTensor *ret) const;

  // negative
  void negative(FixedPointTensor *ret) const;

  // element-wise mul with FixedPointTensor using truncate1
  void mul(const FixedPointTensor *rhs, FixedPointTensor *ret) const;

  // element-wise mul with TensorAdapter
  void mul(const TensorAdapter<T> *rhs, FixedPointTensor *ret) const;

  // div by TensorAdapter
  void div(const TensorAdapter<T> *rhs, FixedPointTensor *ret) const;

  // element-wise mul, use trunc2
  void mul2(const FixedPointTensor *rhs, FixedPointTensor *ret) const;

  // dot_mul
  template <template <typename U, size_t...> class CTensor, size_t... N1>
  void dot_mul(const CTensor<T, N1...> *rhs, FixedPointTensor *ret) const;

  // sum all element
  void sum(FixedPointTensor *ret) const;

  // mat_mul with FixedPointTensor
  void mat_mul(const FixedPointTensor *rhs, FixedPointTensor *ret) const;

  // mat_mul with TensorAdapter
  void mat_mul(const TensorAdapter<T> *rhs, FixedPointTensor *ret) const;

  void exp(FixedPointTensor *ret, size_t iter = 8) const;

  // element-wise relu
  void relu(FixedPointTensor *ret) const;

  // element-wise sigmoid
  void sigmoid(FixedPointTensor *ret) const;

  // softmax axis = -1
  void softmax(FixedPointTensor *ret) const;

  // element-wise polynomial
  void polynomial(const TensorAdapter<T> *coeff, FixedPointTensor *ret) const;

  // element-wise piecewise polynomial
  void polynomial_piecewise(const TensorAdapter<T> *coeff,
                            const TensorAdapter<T> *break_point,
                            FixedPointTensor *ret) const;

  // element-wise compare
  // <
  template <template <typename U, size_t...> class CTensor, size_t... N1>
  void lt(const CTensor<T, N1...> *rhs, BooleanTensor<T> *ret) const;

  // <=
  template <template <typename U, size_t...> class CTensor, size_t... N1>
  void leq(const CTensor<T, N1...> *rhs, BooleanTensor<T> *ret) const;

  // >
  template <template <typename U, size_t...> class CTensor, size_t... N1>
  void gt(const CTensor<T, N1...> *rhs, BooleanTensor<T> *ret) const;

  // >=
  template <template <typename U, size_t...> class CTensor, size_t... N1>
  void geq(const CTensor<T, N1...> *rhs, BooleanTensor<T> *ret) const;

  // ==
  template <template <typename U, size_t...> class CTensor, size_t... N1>
  void eq(const CTensor<T, N1...> *rhs, BooleanTensor<T> *ret) const;

  // !=
  template <template <typename U, size_t...> class CTensor, size_t... N1>
  void neq(const CTensor<T, N1...> *rhs, BooleanTensor<T> *ret) const;

private:
  static inline std::shared_ptr<CircuitContext> aby3_ctx() {
    return paddle::mpc::ContextHolder::mpc_ctx();
  }

  static inline std::shared_ptr<TensorAdapterFactory> tensor_factory() {
    return paddle::mpc::ContextHolder::tensor_factory();
  }

  static void truncate1(FixedPointTensor *op, FixedPointTensor *ret,
                        size_t scaling_factor);

  // reduce last dim
  static void reduce(FixedPointTensor<T, N> *input,
                     FixedPointTensor<T, N> *ret);

  static size_t party() { return aby3_ctx()->party(); }

  static size_t pre_party() { return aby3_ctx()->pre_party(); }

  static size_t next_party() { return aby3_ctx()->next_party(); }

  static void reshare(const TensorAdapter<T> *send_val,
                      TensorAdapter<T> *recv_val) {
    if (party() == 0) {
      aby3_ctx()->network()->template recv(next_party(), *recv_val);
      aby3_ctx()->network()->template send(pre_party(), *send_val);
    } else {
      aby3_ctx()->network()->template send(pre_party(), *send_val);
      aby3_ctx()->network()->template recv(next_party(), *recv_val);
    }
  }

  TensorAdapter<T> *_share[2];
};

} // namespace aby3

#include "fixedpoint_tensor_imp.h"
