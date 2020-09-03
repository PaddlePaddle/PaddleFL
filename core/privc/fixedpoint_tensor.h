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

#include "privc_context.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "../privc3/paddle_tensor.h"
#include "./triplet_generator.h"

namespace privc {

template<typename T>
using TensorAdapter = aby3::TensorAdapter<T>;
using TensorAdapterFactory = aby3::TensorAdapterFactory;

template<size_t N>
inline void fixed64_tensor_mult(const TensorAdapter<int64_t>* lhs,
                                const TensorAdapter<int64_t>* rhs,
                                TensorAdapter<int64_t>* ret) {
    std::transform(lhs->data(), lhs->data() + lhs->numel(),
                   rhs->data(), ret->data(),
                   [] (const int64_t& lhs, const int64_t& rhs) -> int64_t {
                        return fixed64_mult<N>(lhs, rhs);
                     });
}

template<typename T, size_t N>
class FixedPointTensor {

public:
    explicit FixedPointTensor(TensorAdapter<T>* share_tensor);

    ~FixedPointTensor() {};

    template<typename T_>
    class Type2Type {
        typedef T_ type;
    };

    //get mutable shape of tensor
    TensorAdapter<T>* mutable_share();

    const TensorAdapter<T>* share() const;

    size_t numel() const {
        return _share->numel();
    }

    // reveal fixedpointtensor to one party
    void reveal_to_one(size_t party, TensorAdapter<T>* ret) const;

    // reveal fixedpointtensor to all parties
    void reveal(TensorAdapter<T>* ret) const;

    const std::vector<size_t> shape() const;

    //convert TensorAdapter to shares
    static void share(const TensorAdapter<T>* input,
                      TensorAdapter<T>* output_shares[2],
                      block seed = psi::g_zero_block);

    // element-wise add with FixedPointTensor
    void add(const FixedPointTensor* rhs, FixedPointTensor* ret) const;

    // element-wise add with TensorAdapter

    void add(const TensorAdapter<T>* rhs, FixedPointTensor* ret) const;

    // element-wise sub with FixedPointTensor
    void sub(const FixedPointTensor* rhs, FixedPointTensor* ret) const;

    // element-wise sub with TensorAdapter
    void sub(const TensorAdapter<T>* rhs, FixedPointTensor* ret) const;

    // negative
    void negative(FixedPointTensor* ret) const;

    // element-wise mul with FixedPointTensor using truncate1
    void mul(const FixedPointTensor* rhs, FixedPointTensor* ret) const {
      mul_impl<T>(rhs, ret, Type2Type<T>());
    }

    // element-wise mul with TensorAdapter
    void mul(const TensorAdapter<T>* rhs, FixedPointTensor* ret) const {
      mul_impl<T>(rhs, ret, Type2Type<T>());
    }

    // div by TensorAdapter
    void div(const TensorAdapter<T>* rhs, FixedPointTensor* ret) const;

    //sum all element
    void sum(FixedPointTensor* ret) const;

    // mat_mul with FixedPointTensor
    void mat_mul(const FixedPointTensor* rhs, FixedPointTensor* ret) const {
      mat_mul_impl<T>(rhs, ret, Type2Type<T>());
    }

    // mat_mul with TensorAdapter
    void mat_mul(const TensorAdapter<T>* rhs, FixedPointTensor* ret) const;

    // exp approximate: exp(x) = \lim_{n->inf} (1+x/n)^n
    // where n = 2^ite
    // void exp(FixedPointTensor* ret, size_t iter = 8) const;

    // element-wise relu
    void relu(FixedPointTensor* ret) const;

    // element-wise relu with relu'
    // void relu_with_derivative(FixedPointTensor* ret, BooleanTensor<T>* derivative) const;

    // element-wise sigmoid using 3 piecewise polynomials
    void sigmoid(FixedPointTensor* ret) const;

    // softmax axis = -1
    //void softmax(FixedPointTensor* ret) const;

    // element-wise sigmoid using 3 piecewise polynomials
    void argmax(FixedPointTensor* ret) const;

    // element-wise compare
    // <
    template<template<typename U, size_t...> class CTensor,
            size_t... N1>
    void lt(const CTensor<T, N1...>* rhs, CTensor<T, N1...>* ret) const;

    // <=
    template<template<typename U, size_t...> class CTensor,
            size_t... N1>
    void leq(const CTensor<T, N1...>* rhs, CTensor<T, N1...>* ret) const;

    // >
    template<template<typename U, size_t...> class CTensor,
            size_t... N1>
    void gt(const CTensor<T, N1...>* rhs, CTensor<T, N1...>* ret) const;

    // >=
    template<template<typename U, size_t...> class CTensor,
            size_t... N1>
    void geq(const CTensor<T, N1...>* rhs, CTensor<T, N1...>* ret) const;

    // ==
    template<template<typename U, size_t...> class CTensor,
            size_t... N1>
    void eq(const CTensor<T, N1...>* rhs, CTensor<T, N1...>* ret) const;

    // !=
    template<template<typename U, size_t...> class CTensor,
            size_t... N1>
    void neq(const CTensor<T, N1...>* rhs, CTensor<T, N1...>* ret) const;

    // element-wise max
    // if not null, cmp stores true if rhs is bigger
    template<template<typename U, size_t...> class CTensor,
            size_t... N1>
    void max(const CTensor<T, N1...>* rhs,
             FixedPointTensor* ret,
             CTensor<T, N1...>* cmp = nullptr) const;

private:
    static inline std::shared_ptr<AbstractContext> privc_ctx() {
      return paddle::mpc::ContextHolder::mpc_ctx();
    }

    static inline std::shared_ptr<TensorAdapterFactory> tensor_factory() {
        return paddle::mpc::ContextHolder::tensor_factory();
    }

    static inline std::shared_ptr<TripletGenerator<T, N>> tripletor() {
        return std::dynamic_pointer_cast<PrivCContext>(privc_ctx())->triplet_generator();
    }

    static size_t party() {
        return privc_ctx()->party();
    }

    static size_t next_party() {
        return privc_ctx()->next_party();
    }
    static inline AbstractNetwork* net() {
      return privc_ctx()->network();
    }

    // mul_impl with FixedPointTensor
    template<typename T_>
    void mul_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor mul is not implemented yet");
    }
    template<typename T_>
    void mul_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<int64_t>) const;

    // mul_impl with TensorAdapter
    template<typename T_>
    void mul_impl(const TensorAdapter<T>* rhs, FixedPointTensor* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor mul is not implemented yet");
    }
    template<typename T_>
    void mul_impl(const TensorAdapter<T>* rhs, FixedPointTensor* ret, Type2Type<int64_t>) const;

    // mat_mul_impl with FixedPointTensor
    template<typename T_>
    void mat_mul_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor mul is not implemented yet");
    }
    template<typename T_>
    void mat_mul_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<int64_t>) const;

    TensorAdapter<T>* _share;

};

} //namespace privc

#include "fixedpoint_tensor_imp.h"
