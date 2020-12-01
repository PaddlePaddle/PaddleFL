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

#include <memory>
#include <string>
#include <vector>

#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/common/tensor_adapter.h"

template<typename T>
using TensorAdapter = common::TensorAdapter<T>;
using TensorAdapterFactory = common::TensorAdapterFactory;

namespace aby3 {

template<typename T, size_t N>
class FixedPointTensor;

template<typename T>
class BooleanTensor {

public:
    BooleanTensor(TensorAdapter<T>* share_tensor[2]);

    BooleanTensor(TensorAdapter<T>* tensor0, TensorAdapter<T>* tensor1);

    BooleanTensor();

    // ABY3 a2b
    template<size_t N>
    BooleanTensor& operator=(const FixedPointTensor<T, N>* other);

    ~BooleanTensor() {}

    //get share
    TensorAdapter<T>* share(size_t idx);

    const TensorAdapter<T>* share(size_t idx) const;

    // reveal boolean tensor to one party
    void reveal_to_one(size_t party_num, TensorAdapter<T>* ret) const;

    // reveal boolean tensor to all parties
    void reveal(TensorAdapter<T>* ret) const;

    const std::vector<size_t> shape() const;

    size_t numel() const;

    // //convert TensorAdapter to shares
    // static void share(const TensorAdapter<T>* input,
    //                   TensorAdapter<T>* output_shares[3],
    //                   const std::string& rnd_seed = "");

    // element-wise xor with BooleanTensor
    void bitwise_xor(const BooleanTensor* rhs, BooleanTensor* ret) const;

    // element-wise xor with TensorAdapter
    void bitwise_xor(const TensorAdapter<T>* rhs, BooleanTensor* ret) const;

    // element-wise and with BooleanTensor
    void bitwise_and(const BooleanTensor* rhs, BooleanTensor* ret) const;

    // element-wise and with TensorAdapter
    void bitwise_and(const TensorAdapter<T>* rhs, BooleanTensor* ret) const;

    // element-wise or
    // for both tensor adapter and boolean tensor
    template<template<typename U> class CTensor>
    void bitwise_or(const CTensor<T>* rhs, BooleanTensor* ret) const;

    // element-wise not
    void bitwise_not(BooleanTensor* ret) const;

    // element-wise lshift
    void lshift(size_t rhs, BooleanTensor* ret) const;

    // element-wise rshift
    void rshift(size_t rhs, BooleanTensor* ret) const;

    // element-wise logical_rshift
    void logical_rshift(size_t rhs, BooleanTensor* ret) const;

    // element-wise ppa with BooleanTensor
    void ppa(const BooleanTensor* rhs, BooleanTensor*ret , size_t nbits) const;

    // ABY3 b2a
    template<size_t N>
    void b2a(FixedPointTensor<T, N>* ret) const;

    // ABY3 ab mul
    // this is an one-bit boolean share
    template<size_t N>
    void mul(const TensorAdapter<T>* rhs, FixedPointTensor<T, N>* ret, size_t rhs_party) const;

    // ABY3 ab mul
    // this is an one-bit boolean share
    template<size_t N>
    void mul(const FixedPointTensor<T, N>* rhs, FixedPointTensor<T, N>* ret) const;

    // extract to this
    template<size_t N>
    void bit_extract(size_t i, const FixedPointTensor<T, N>* in);

    // extract from this to ret
    void bit_extract(size_t i, BooleanTensor* ret) const;

    // turn all 1s to 0s except the last 1 in a col
    // given cmp result from max pooling, generate one hot tensor
    // indicating which element is max
    // inplace transform
    void onehot_from_cmp();

private:
  static inline std::shared_ptr<AbstractContext> aby3_ctx() {
    return paddle::mpc::ContextHolder::mpc_ctx();
  }

    static inline std::shared_ptr<TensorAdapterFactory> tensor_factory() {
        return paddle::mpc::ContextHolder::tensor_factory();
    }

    size_t pre_party() const;

    size_t next_party() const;

    size_t party() const;

private:
    TensorAdapter<T>* _share[2];

};

} //namespace aby3

#include "boolean_tensor_impl.h"
