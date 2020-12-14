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

#include "core/privc/privc_context.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "../common/paddle_tensor.h"
#include "./triplet_generator.h"
#include "core/common/tensor_adapter_factory.h"
#include "core/privc/triplet_generator.h"
#include "core/privc/utils.h"

namespace privc {

template<typename T>
using TensorAdapter = common::TensorAdapter<T>;
using TensorAdapterFactory = common::TensorAdapterFactory;

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
                      block seed = common::g_zero_block);

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

    // exp
    void exp(FixedPointTensor<T, N>* ret, size_t iter = 8) const;

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

    // div by FixedPointTensor using gc div
    void long_div(const FixedPointTensor* rhs, FixedPointTensor* ret) const {
      long_div_impl<T>(rhs, ret, Type2Type<T>());
    }

    //sum all element
    void sum(FixedPointTensor* ret) const;

    //reduce last dim
    void reduce(FixedPointTensor* ret) const;

    // mat_mul with FixedPointTensor
    void mat_mul(const FixedPointTensor* rhs, FixedPointTensor* ret) const {
      mat_mul_impl<T>(rhs, ret, Type2Type<T>());
    }

    // mat_mul with TensorAdapter
    void mat_mul(const TensorAdapter<T>* rhs, FixedPointTensor* ret) const;

    // element-wise relu
    void relu(FixedPointTensor* ret) const {
      relu_impl<T>(ret, Type2Type<T>());
    }

    // element-wise sigmoid
    void sigmoid(FixedPointTensor* ret) const {
      sigmoid_impl<T>(ret, Type2Type<T>());
    }

    // element-wise softmax
    void softmax(FixedPointTensor* ret, bool use_relu = false) const {
      softmax_impl<T>(ret, use_relu, Type2Type<T>());
    }

    // matrix argmax
    // return max index in one-hot
    void argmax(FixedPointTensor<T, N>* ret) const {
      argmax_impl<T>(ret, Type2Type<T>());
    }

private:
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

    // long_div_impl with FixedPointTensor
    template<typename T_>
    void long_div_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor long div is not implemented yet");
    }
    template<typename T_>
    void long_div_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<int64_t>) const;

    // mat_mul_impl with FixedPointTensor
    template<typename T_>
    void mat_mul_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor mat mul is not implemented yet");
    }
    template<typename T_>
    void mat_mul_impl(const FixedPointTensor* rhs, FixedPointTensor* ret, Type2Type<int64_t>) const;

    // relu_impl with FixedPointTensor
    template<typename T_>
    void relu_impl(FixedPointTensor* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor relu is not implemented yet");
    }
    template<typename T_>
    void relu_impl(FixedPointTensor* ret, Type2Type<int64_t>) const;

    // sigmoid_impl with FixedPointTensor
    template<typename T_>
    void sigmoid_impl(FixedPointTensor* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor sigmoid is not implemented yet");
    }
    template<typename T_>
    void sigmoid_impl(FixedPointTensor* ret, Type2Type<int64_t>) const;

    // argmax_impl with FixedPointTensor
    template<typename T_>
    void argmax_impl(FixedPointTensor<T, N>* ret, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor argmax is not implemented yet");
    }
    template<typename T_>
    void argmax_impl(FixedPointTensor<T, N>* ret, Type2Type<int64_t>) const;

    // softmax_impl with FixedPointTensor
    template<typename T_>
    void softmax_impl(FixedPointTensor<T, N>* ret, bool use_relu, Type2Type<T_>) const {
      PADDLE_THROW("type except `int64_t` for fixedtensor softmax is not implemented yet");
    }
    template<typename T_>
    void softmax_impl(FixedPointTensor<T, N>* ret, bool use_relu, Type2Type<int64_t>) const;

    TensorAdapter<T>* _share;

};

} //namespace privc

#include "fixedpoint_tensor_imp.h"
