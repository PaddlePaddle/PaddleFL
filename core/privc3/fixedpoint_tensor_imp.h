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
#include <algorithm>

#include "paddle/fluid/platform/enforce.h"
#include "prng.h"

namespace aby3 {
template<typename T, size_t N>
FixedPointTensor<T, N>::FixedPointTensor(TensorAdapter<T>* share_tensor[2]) {
    // TODO: check tensors' shapes
    _share[0] = share_tensor[0];
    _share[1] = share_tensor[1];
}

template<typename T, size_t N>
FixedPointTensor<T, N>::FixedPointTensor(TensorAdapter<T>* share_tensor_0,
                                         TensorAdapter<T>* share_tensor_1) {
    // TODO: check tensors' shapes
    _share[0] = share_tensor_0;
    _share[1] = share_tensor_1;
}

template<typename T, size_t N>
TensorAdapter<T>* FixedPointTensor<T, N>::mutable_share(size_t idx) {
    PADDLE_ENFORCE_LT(idx, 2, "Input should be less than 2.");
    return _share[idx];
}

template<typename T, size_t N>
const TensorAdapter<T>* FixedPointTensor<T, N>::share(size_t idx) const {
    PADDLE_ENFORCE_LT(idx, 2, "Input should be less than 2.");
    return _share[idx];
}

// reveal fixedpointtensor to one party
template<typename T, size_t N>
void FixedPointTensor<T, N>::reveal_to_one(size_t party,
                                           TensorAdapter<T>* ret) const {

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
template<typename T, size_t N>
void FixedPointTensor<T, N>::reveal(TensorAdapter<T>* ret) const {
    for (size_t i = 0; i < 3; ++i) {
        reveal_to_one(i, ret);
    }
}

template<typename T, size_t N>
const std::vector<size_t> FixedPointTensor<T, N>::shape() const {
    return _share[0]->shape();
}

//convert TensorAdapter to shares
template<typename T, size_t N>
void FixedPointTensor<T, N>::share(const TensorAdapter<T>* input,
                                    TensorAdapter<T>* output_shares[3],
                                    block seed) {

    if (equals(seed, g_zero_block)) {
        seed = block_from_dev_urandom();
    }
    //set seed of prng[2]
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

template<typename T, size_t N>
void FixedPointTensor<T, N>::add(const FixedPointTensor<T, N>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    _share[0]->add(rhs->_share[0], ret->_share[0]);
    _share[1]->add(rhs->_share[1], ret->_share[1]);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::add(const TensorAdapter<T>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
                        "no match scaling factor");
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

template<typename T, size_t N>
void FixedPointTensor<T, N>::sub(const FixedPointTensor<T, N>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    _share[0]->sub(rhs->_share[0], ret->_share[0]);
    _share[1]->sub(rhs->_share[1], ret->_share[1]);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sub(const TensorAdapter<T>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
                        "no match scaling factor");
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

template<typename T, size_t N>
void FixedPointTensor<T, N>::negative(FixedPointTensor<T, N>* ret) const {
    _share[0]->negative(ret->_share[0]);
    _share[1]->negative(ret->_share[1]);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mul(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret) const {
    mul_trunc(this, rhs, ret, &TensorAdapter<T>::mul);
}

#ifdef USE_ABY3_TRUNC1 //use aby3 trunc1
template<typename T, size_t N>
void FixedPointTensor<T, N>::truncate(const FixedPointTensor<T, N>* op,
                                       FixedPointTensor<T, N>* ret,
                                       size_t scaling_factor) {
    if (scaling_factor == 0) {
        op->share(0)->copy(ret->mutable_share(0));
        op->share(1)->copy(ret->mutable_share(1));
    }
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
        // trunc from [SecureML, Thm.1]
        ret->_share[0]->negative(ret->_share[0]);
        ret->_share[0]->rshift(scaling_factor, ret->_share[0]);
        ret->_share[0]->negative(ret->_share[0]);
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

#else // use truncate3

// Protocol. `truncate3` (illustrated for data type T = int64_t)
// motivation:
// truncates in aby3 may cause msb error with small probability
// the reason is that before rishft op, its masked value e.g., x' - r' may overflow in int64_t
// so that, in `truncate3`, we limit r' in (-2^62, 2^62) to avoid the problem.

// notice:
// when r' is contrainted in (-2^62, 2^62),
// the SD (statistical distance) of x' - r' between this
// and r' in Z_{2^64} is equal to |X| / (2^63 + |X|)

// detail protocol:
// P2 randomly generates r' \in (-2^62, 2^62), randomly generates r'_0, r_0, r_1 in Z_{2^64},
// P2 compute r'_1 = r' - r'_0, r_2 = r'/2^N - r_0 - r_1, let x2 = r_2
// P2 send r_0, r'_0 to P0, send r_1, r'_1 to P1
// P1 and P0 execute "reveal x - r' to P1"
// P1 compute x1 = (x - r') / 2^N + r_1
// P0 set x0 = r_0
// P0, P1, P2 invoke reshare() with inputs x0, x1, x2 respectively.
template<typename T, size_t N>
void FixedPointTensor<T, N>::truncate(const FixedPointTensor<T, N>* op,
                                       FixedPointTensor<T, N>* ret,
                                       size_t scaling_factor) {
    if (scaling_factor == 0) {
        op->share(0)->copy(ret->mutable_share(0));
        op->share(1)->copy(ret->mutable_share(1));
        return;
    }
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    if (party() == 2) {
        for (int i = 0; i < 7; ++i) {
            temp.emplace_back(
                tensor_factory()->template create<T>(op->shape()));
        }
        // r'
        aby3_ctx()->template gen_random_private(*temp[0]);
        temp[0]->rshift(1, temp[0].get());

        //r'_0, r'_1
        aby3_ctx()->template gen_random_private(*temp[1]);
        temp[0]->sub(temp[1].get(), temp[2].get());
        // r, r_0, r_1
        temp[0]->rshift(scaling_factor, temp[3].get());
        aby3_ctx()->template gen_random_private(*temp[4]);
        aby3_ctx()->template gen_random_private(*temp[5]);
        // r_2
        temp[3]->sub(temp[4].get(), temp[6].get());
        temp[6]->sub(temp[5].get(), temp[6].get());

        aby3_ctx()->network()->template send(1, *temp[2]);
        aby3_ctx()->network()->template send(1, *temp[5]);
        aby3_ctx()->network()->template send(0, *temp[1]);
        aby3_ctx()->network()->template send(0, *temp[4]);

        temp[6]->copy(ret->mutable_share(0));

    } else if (party() == 1) {
        for (int i = 0; i < 4; ++i) {
            temp.emplace_back(
                tensor_factory()->template create<T>(op->shape()));
        }
        // r'_1, r_1
        aby3_ctx()->network()->template recv(2, *temp[0]);
        aby3_ctx()->network()->template recv(2, *temp[1]);
        // recv x0 - r'_0 from party 0
        aby3_ctx()->network()->template recv(0, *temp[2]);
        //reveal x - r' to party 1
        op->share(0)->add(op->share(1), temp[3].get());
        temp[3]->add(temp[2].get(), temp[3].get());
        temp[3]->sub(temp[0].get(), temp[3].get());
        // truncate x-r'
        temp[3]->rshift(scaling_factor, temp[3].get());

        temp[3]->add(temp[1].get(), ret->mutable_share(0));
    } else {
        for (int i = 0; i < 3; ++i) {
            temp.emplace_back(
                tensor_factory()->template create<T>(op->shape()));
        }
        // r'_0, r_0
        aby3_ctx()->network()->template recv(2, *temp[0]);
        aby3_ctx()->network()->template recv(2, *temp[1]);
        //send x0 - r'_0 to party 1
        op->share(0)->sub(temp[0].get(), temp[2].get());
        aby3_ctx()->network()->template send(1, *temp[2]);
        temp[1]->copy(ret->mutable_share(0));
    }

    reshare(ret->share(0), ret->mutable_share(1));

    // compensation for carry in
    auto tensor_carry_in = tensor_factory()->template create<T>(ret->shape());
    assign_to_tensor(tensor_carry_in.get(), (T)1);
    tensor_carry_in->scaling_factor() = N;
    ret->add(tensor_carry_in.get(), ret);
}
#endif //USE_ABY3_TRUNC1

template<typename T, size_t N>
template<typename MulFunc>
void FixedPointTensor<T, N>::mul_trunc(const FixedPointTensor<T, N>* lhs,
                                        const FixedPointTensor<T, N>* rhs,
                                        FixedPointTensor<T, N>* ret,
                                        MulFunc mul_func) {

    auto r_zero = tensor_factory()->template create<T>(ret->shape());
    aby3_ctx()->gen_zero_sharing_arithmetic(*r_zero.get());

    // temp = _share[0]->mul(rhs->_share[0]) +
    //        _share[0]->mul(rhs->_share[1]) +
    //        _share[1]->mul(rhs->_share[0]) +
    //        r_zero
    auto temp = tensor_factory()->template create<T>(ret->shape());
    auto temp1 = tensor_factory()->template create<T>(ret->shape());

    // use mul_func to fit both element_wise mul and mat mul
    (lhs->share(0)->*mul_func)(rhs->share(0), temp.get());
    (lhs->share(0)->*mul_func)(rhs->share(1), temp1.get());
    temp1->add(temp.get(), temp1.get());

    (lhs->share(1)->*mul_func)(rhs->share(0), temp.get());
    temp1->add(r_zero.get(), temp1.get());
    temp->add(temp1.get(), temp.get());

    auto temp2 = tensor_factory()->template create<T>(ret->shape());
    auto temp3 = tensor_factory()->template create<T>(ret->shape());

    TensorAdapter<int64_t>* temp_array[2] = {temp2.get(), temp3.get()};

    std::shared_ptr<FixedPointTensor<T, N>> ret_no_trunc =
            std::make_shared<FixedPointTensor<T, N>>(temp_array);

    temp->copy(ret_no_trunc->_share[0]);
    reshare(temp.get(), ret_no_trunc->_share[1]);

    truncate(ret_no_trunc.get(), ret, N);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mul(const TensorAdapter<T>* rhs,
                                 FixedPointTensor<T, N>* ret) const {
    // PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
    //                   "no match scaling factor");
    auto temp0 = tensor_factory()->template create<T>(this->shape());
    auto temp1 = tensor_factory()->template create<T>(this->shape());
    std::shared_ptr<FixedPointTensor<T, N>> temp =
        std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());

    _share[0]->mul(rhs, temp->_share[0]);
    _share[1]->mul(rhs, temp->_share[1]);
    truncate(temp.get(), ret, rhs->scaling_factor());
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sum(FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");
    T sum1 = (T) 0;
    T sum2 = (T) 0;
    T* iter_0 = _share[0]->data();
    T* iter_1 = _share[1]->data();
    for (int i = 0; i < this->numel(); ++i) {
        sum1 += *(iter_0 + i);
        sum2 += *(iter_1 + i);
    }
    assign_to_tensor(ret->_share[0], sum1);
    assign_to_tensor(ret->_share[1], sum2);
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::dot_mul(const CTensor<T, N1...>* rhs,
                                     FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");

    auto temp0 = tensor_factory()->template create<T>(this->shape());
    auto temp1 = tensor_factory()->template create<T>(this->shape());
    std::shared_ptr<FixedPointTensor<T, N>> temp =
            std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
    this->mul(rhs, temp.get());
    temp->sum(ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mat_mul(const FixedPointTensor<T, N>* rhs,
                                     FixedPointTensor<T, N>* ret) const {
    mul_trunc(this, rhs, ret, &TensorAdapter<T>::mat_mul);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::mat_mul(const TensorAdapter<T>* rhs,
                                     FixedPointTensor<T, N>* ret) const {
    _share[0]->mat_mul(rhs, ret->_share[0]);
    _share[1]->mat_mul(rhs, ret->_share[1]);
    truncate(ret, ret, rhs->scaling_factor());
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::div(const TensorAdapter<T>* rhs,
                                 FixedPointTensor<T, N>* ret) const {
    PADDLE_ENFORCE_EQ(N, rhs->scaling_factor(),
                        "no match scaling factor");

    auto temp = tensor_factory()->template create<T>(this->shape());

    double scale = std::pow(2, rhs->scaling_factor());
    auto inverse = [scale](T d) -> T {
                    return 1.0 * scale / d * scale; };
    std::transform(rhs->data(), rhs->data() + rhs->numel(),
                                temp->data(), inverse);
    temp->scaling_factor() = rhs->scaling_factor();

    this->mul(temp.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::div(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret,
                                 size_t iter, double x0) const {
    auto temp0 = tensor_factory()->template create<T>(ret->shape());
    auto temp1 = tensor_factory()->template create<T>(ret->shape());
    std::shared_ptr<FixedPointTensor<T, N>> temp =
        std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
    reciprocal(rhs, temp.get(), iter, x0);
    this->mul(temp.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::exp(FixedPointTensor<T, N>* ret,
                                 size_t iter) const {
    // exp approximate: exp(x) = \lim_{n->inf} (1+x/n)^n
    // where n = 2^ite
    auto pow_iter = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(pow_iter.get(), (T) (pow(2, N -iter)));
    pow_iter->scaling_factor() = N;

    auto tensor_one = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(tensor_one.get(), (T) 1 << N);
    tensor_one->scaling_factor() = N;

    this->mul(pow_iter.get(), ret);

    ret->add(tensor_one.get(), ret);

    for (int i = 0; i < iter; ++i) {
        ret->mul(ret, ret);
    }
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::relu(FixedPointTensor<T, N>* ret) const {
    //utilize polynomial_piecewise
    // break_point = {0}, coeff[0] = {0, 0}, coeff[1] = {0, 1}
    // break_point.shape = {1, this->shape}, coeff.shape = {2, 2, this->shape}

    auto shape_ = shape();
    //construct break_point
    auto b_shape = shape_;
    b_shape.insert(b_shape.begin(), 1);

    auto break_point = tensor_factory()->template create<T>(b_shape);

    T* b_ptr = break_point->data();
    for (size_t i = 0; i < break_point->numel(); ++i) {
        b_ptr[i] = 0;
    }
    break_point->scaling_factor() = N;

    //contruct coeff
    std::vector<size_t> c_shape = {2, 2};
    c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());

    auto coeff = tensor_factory()->template create<T>(c_shape);

    T* c_ptr = coeff->data();

    for (size_t i = 0; i < 3 * this->numel(); ++i) {
        c_ptr[i] = 0;
    }
    for (size_t i = 3 * this->numel(); i < 4 * this->numel(); ++i) {
        c_ptr[i] = (T) 1 << N;
    }
    coeff->scaling_factor() = N;

    this->polynomial_piecewise(coeff.get(), break_point.get(), ret);
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::relu_with_derivative(
    FixedPointTensor<T, N>* ret, BooleanTensor<T>* derivative) const {

    auto shape_ = shape();
    auto zero = tensor_factory()->template create<T>(shape_);

    assign_to_tensor(zero.get(), (T)0);
    zero->scaling_factor() = N;

    auto tmp0 = tensor_factory()->template create<T>(shape_);
    auto tmp1 = tensor_factory()->template create<T>(shape_);

    BooleanTensor<T> der(tmp0.get(), tmp1.get());

    gt(zero.get(), &der);

    der.mul(this, ret);

    if (derivative) {
        der.share(0)->copy(derivative->share(0));
        der.share(1)->copy(derivative->share(1));
    }
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid_chebyshev(FixedPointTensor<T, N>* ret) const {
    //utilize Chebyshev polynomial approximation
    // more accurate in small range, such as [-4, 4]
    auto shape = ret->shape();
    std::vector<size_t> shape_ = shape;
    shape_.insert(shape_.begin(), 10);
    auto numel = ret->numel();
    auto coeff = tensor_factory()->template create<T>(shape_);
    std::vector<double> w;
    w.resize(10, 0.0f);
    w[0] = 0.5;
    w[1] = 0.2159198015;
    w[3] = -0.0082176259;
    w[5] = 0.0001825597;
    w[7] = -0.0000018848;
    w[9] = 0.0000000072;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < numel; ++j) {
            *(coeff->data() + i * numel + j) = (T) (w[i] * pow(2, N));
        }
    }
    coeff->scaling_factor() = N;
    polynomial(coeff.get(), ret);
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid(FixedPointTensor<T, N>* ret) const {
    //utilize polynomial_piecewise
    // break_point = {-2.5, 2.5}
    // coeff[0] = {10^-4, 0}, coeff[1] = {0.5, 0.17}
    // coeff[2] = {1 - 10^-4, 0}
    // break_point.shape = {2, this->shape}, coeff.shape = {3, 2, this->shape}

    //construct break_point
    auto shape_ = shape();
    //construct break_point
    auto b_shape = shape_;
    b_shape.insert(b_shape.begin(), 2);

    auto break_point = tensor_factory()->template create<T>(b_shape);

    T* b_ptr = break_point->data();
    for (size_t i = 0; i < break_point->numel(); ++i) {
        b_ptr[i] = 0;
    }
    for (size_t i = 0; i < break_point->numel() / 2; ++i) {
        b_ptr[i] = (T) (-2.5 * pow(2, N));
    }
    for (size_t i = break_point->numel() / 2; i < break_point->numel(); ++i) {
        b_ptr[i] = (T) (2.5 * pow(2, N));
    }
    break_point->scaling_factor() = N;

    //contruct coeff
    std::vector<size_t> c_shape = {3, 2};
    c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());

    auto coeff = tensor_factory()->template create<T>(c_shape);

    T* c_ptr = coeff->data();

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

template< typename T, size_t N>
void FixedPointTensor<T, N>::sigmoid_enhanced(FixedPointTensor<T, N>* ret) const {
    //utilize polynomial_piecewise
    // break_point = {-5, -2.5, 2.5, 5}
    // coeff[0] = {10^-4, 0}, coeff[1] = {0.145, 0.02776}
    // coeff[2] = {0.5, 0.17}, coeff[3] = {0.85498, 0.02776}, coeff[4] = {0.9999, 0}
    // break_point.shape = {4, this->shape}, coeff.shape = {5, 2, this->shape}

    //construct break_point
    auto shape_ = shape();
    //construct break_point
    auto b_shape = shape_;
    b_shape.insert(b_shape.begin(), 4);

    auto break_point = tensor_factory()->template create<T>(b_shape);

    T* b_ptr = break_point->data();
    auto numel = ret->numel();
    double scale = std::pow(2, N);
    for (size_t i = 0; i < numel; ++i) {
        b_ptr[i] = (T) (-5 * scale);
        b_ptr[i + numel] = (T) (-2.5 * scale);
        b_ptr[i + 2 * numel] = (T) (2.5 * scale);
        b_ptr[i + 3 * numel] = (T) (5 * scale);
    }
    break_point->scaling_factor() = N;

    //contruct coeff
    std::vector<size_t> c_shape = {5, 2};
    c_shape.insert(c_shape.end(), shape_.begin(), shape_.end());
    auto coeff = tensor_factory()->template create<T>(c_shape);
    T* c_ptr = coeff->data();
    for (size_t i = 0; i < numel; ++i) {
        c_ptr[i] = 0.0001 * scale;
        c_ptr[i + numel] = 0;
        c_ptr[i + 2 * numel] = 0.145 * scale;
        c_ptr[i + 3 * numel] = 0.02776 * scale;
        c_ptr[i + 4 * numel] = 0.5 * scale;
        c_ptr[i + 5 * numel] = 0.17 * scale;
        c_ptr[i + 6 * numel] = 0.85498 * scale;
        c_ptr[i + 7 * numel] = 0.02776 * scale;
        c_ptr[i + 8 * numel] = 0.9999 * scale;
        c_ptr[i + 9 * numel] = 0 * scale;
    }
    coeff->scaling_factor() = N;

    this->polynomial_piecewise(coeff.get(), break_point.get(), ret);
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::softmax(FixedPointTensor<T, N>* ret,
                                     bool use_relu, bool use_long_div) const {
    // softmax axis = -1
    const size_t col = *(shape().end() - 1);
    const size_t row = numel() / col;

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    // 11 for allocating temp tensor
    for (size_t i = 0; i < 11; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }

    temp[0]->reshape({row, col});
    temp[1]->reshape({row, col});
    FixedPointTensor<T, N> x(temp[0].get(), temp[1].get());

    if (!use_relu) {
        temp[2]->reshape({col, row});
        temp[3]->reshape({col, row});

        temp[4]->reshape({1, row});
        temp[5]->reshape({1, row});
    }
    FixedPointTensor<T, N> x_t(temp[2].get(), temp[3].get());
    FixedPointTensor<T, N> max_x_t(temp[4].get(), temp[5].get());

    temp[6]->reshape({row, 1});
    temp[7]->reshape({row, 1});
    FixedPointTensor<T, N> max_x(temp[6].get(), temp[7].get());

    temp[8]->reshape({row, col});
    temp[9]->reshape({row, col});
    FixedPointTensor<T, N> max_x_broadcast(temp[8].get(), temp[9].get());

    temp[10]->reshape({row, col});
    auto exp_lower_bound = temp[10].get();

    auto transpose = [](const TensorAdapter<T>* in, TensorAdapter<T>* out) {
        // suppose input dims = 2
        const size_t col = in->shape()[1];
        const size_t row = in->shape()[0];
        const size_t numel = in->numel();

        for (size_t k = 0; k < numel; ++k) {
            size_t i = k / row;
            size_t j = k % row;
            out->data()[k] = in->data()[j * col + i];
        }
    };

    auto broadcast = [](const TensorAdapter<T>* in, TensorAdapter<T>* out) {
        // suppose input dims = 2
        // in shape = [row, 1]
        const size_t col = out->shape()[1];
        const size_t row = out->shape()[0];
        for (size_t k = 0; k < out->numel(); ++k) {
            size_t i = k / col;
            out->data()[k] = in->data()[i];
        }
    };

    share(0)->copy(x.mutable_share(0));
    share(1)->copy(x.mutable_share(1));

    if (use_relu) {

        x.relu(&x);

    } else { // use exp
        transpose(x.share(0), x_t.mutable_share(0));
        transpose(x.share(1), x_t.mutable_share(1));

        // x = max(input - max(input), exp_lower_bound)
        x_t.max_pooling(&max_x_t);

        transpose(max_x_t.share(0), max_x.mutable_share(0));
        transpose(max_x_t.share(1), max_x.mutable_share(1));

        broadcast(max_x.share(0), max_x_broadcast.mutable_share(0));
        broadcast(max_x.share(1), max_x_broadcast.mutable_share(1));

        x.sub(&max_x_broadcast, &x);

        // n = 64, see exp
        assign_to_tensor(exp_lower_bound, (T)(-64 * (1 << N)));
        exp_lower_bound->scaling_factor() = N;

        x.sub(exp_lower_bound, &x);
        x.relu(&x);
        x.add(exp_lower_bound, &x);

        x.exp(&x);
    }

    // reuse max_x as sum
    reduce(&x, &max_x);

    if (!use_long_div) { // invert sum by Newton's method
    // divisor range = [1/col, 1.0]
    // TODO: find better iter num & init val
        reciprocal(&max_x, &max_x, 16, 0.5 / col);
    }

    broadcast(max_x.share(0), max_x_broadcast.mutable_share(0));
    broadcast(max_x.share(1), max_x_broadcast.mutable_share(1));

    if (use_long_div) {
        x.long_div(&max_x_broadcast, &x, 1);
    } else {
        x.mul(&max_x_broadcast, &x);
    }

    x.share(0)->copy(ret->mutable_share(0));
    x.share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::long_div(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret,
                                 size_t int_len) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 16; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(ret->shape()));
    }

    BooleanTensor<T> sign_lhs(temp[0].get(), temp[1].get());
    BooleanTensor<T> sign_rhs(temp[2].get(), temp[3].get());
    BooleanTensor<T> sign_ret(temp[4].get(), temp[5].get());
    FixedPointTensor<T, N> abs_lhs(temp[6].get(), temp[7].get());
    FixedPointTensor<T, N> abs_rhs(temp[8].get(), temp[9].get());
    FixedPointTensor<T, N> sub_rhs(temp[10].get(), temp[11].get());
    BooleanTensor<T> cmp_res(temp[12].get(), temp[13].get());
    BooleanTensor<T> cmp_res_all(temp[14].get(), temp[15].get());

    assign_to_tensor(cmp_res_all.share(0), (T)0);
    assign_to_tensor(cmp_res_all.share(1), (T)0);

    const size_t msb = sizeof(T) * 8 - 1;
    sign_lhs.bit_extract(msb, this);
    sign_rhs.bit_extract(msb, rhs);
    sign_lhs.bitwise_xor(&sign_rhs, &sign_ret);

    auto lshift = []  (const FixedPointTensor<T, N>* in,
                       size_t rhs,
                       FixedPointTensor<T, N>* out) {
        in->share(0)->lshift(rhs, out->mutable_share(0));
        in->share(1)->lshift(rhs, out->mutable_share(1));
    };

    // abs = val - 2 * sign * val
    auto abs = [lshift] (const FixedPointTensor<T, N>* in,
                   const BooleanTensor<T>* sign,
                   FixedPointTensor<T, N>* out) {
        lshift(in, 1, out);
        sign->mul(out, out);
        in->sub(out, out);
    };

    auto out0 = tensor_factory()->template create<T>(ret->shape());

    abs(this, &sign_lhs, &abs_lhs);

    abs(rhs, &sign_rhs, &abs_rhs);


    for (ssize_t i = int_len - 1; i >= 0; --i) {
        lshift(&abs_rhs, i, &sub_rhs);


        abs_lhs.gt(&sub_rhs, &cmp_res);


        cmp_res.mul(&sub_rhs, &sub_rhs);
        cmp_res.lshift(N + i, &cmp_res);
        abs_lhs.sub(&sub_rhs, &abs_lhs);
        cmp_res.bitwise_xor(&cmp_res_all, &cmp_res_all);

    }

    for (size_t i = 1; i <= N; ++i) {
        truncate(&abs_rhs, &sub_rhs, i);
        abs_lhs.gt(&sub_rhs, &cmp_res);
        cmp_res.mul(&sub_rhs, &sub_rhs);
        cmp_res.lshift(N - i, &cmp_res);
        abs_lhs.sub(&sub_rhs, &abs_lhs);
        cmp_res.bitwise_xor(&cmp_res_all, &cmp_res_all);
    }

    // use abs_lhs as buffer
    cmp_res_all.b2a(&abs_lhs);

    abs(&abs_lhs, &sign_ret, ret);
}

// reduce last dim
template <typename T, size_t N>
void FixedPointTensor<T, N>::reduce(FixedPointTensor<T, N>* input,
                                    FixedPointTensor<T, N>* ret) {
    //enfoce shape: input->shape[0 ... (n-2)] == ret shape
    auto& shape = input->shape();
    size_t ite_size = shape[shape.size() - 1];

    T* ret_begin_ptr_0 = ret->_share[0]->data();
    T* ret_begin_ptr_1 = ret->_share[1]->data();

    T* input_begin_ptr_0 = input->_share[0]->data();
    T* input_begin_ptr_1 = input->_share[1]->data();

    for (int j = 0; j < ret->numel(); ++j) {
        *(ret_begin_ptr_0 + j) = *(input_begin_ptr_0 + j * ite_size);
        *(ret_begin_ptr_1 + j) = *(input_begin_ptr_1 + j * ite_size);
        for (int i =  1; i < ite_size; ++i) {
            *(ret_begin_ptr_0 + j) +=
                        *(input_begin_ptr_0 + j * ite_size + i);
            *(ret_begin_ptr_1 + j) +=
                        *(input_begin_ptr_1 + j * ite_size + i);
        }
    }
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::polynomial(const TensorAdapter<T>* coeff,
                                        FixedPointTensor<T, N>* ret) const {

    // e.g., x.shape = {2, 3}, coeff.shape = {n, 2, 3} (n: polynomial power)

    //TODO: improve performance: [ABY3]
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 7; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> x_pow_i =
            std::make_shared<FixedPointTensor<T, N>>(
                                temp[0].get(), temp[1].get());
    std::shared_ptr<FixedPointTensor<T, N>> temp_fixed =
            std::make_shared<FixedPointTensor<T, N>>(
                                temp[2].get(), temp[3].get());
    std::shared_ptr<FixedPointTensor<T, N>> result =
            std::make_shared<FixedPointTensor<T, N>>(
                                temp[5].get(), temp[6].get());
    assign_to_tensor(result->_share[0], (T) 0);
    assign_to_tensor(result->_share[1], (T) 0);

    //x_pow_i.get() = 1;
    assign_to_tensor(x_pow_i.get()->_share[0], (T) 0);
    assign_to_tensor(x_pow_i.get()->_share[1], (T) 0);
    assign_to_tensor(temp[4].get(), (T) 1 << N);
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
        result->add(temp_fixed.get(), result.get());
        x_pow_i->mul(this, x_pow_i.get());
    }
    result->share(0)->copy(ret->mutable_share(0));
    result->share(1)->copy(ret->mutable_share(1));
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::polynomial_piecewise(
                    const TensorAdapter<T>* coeff,
                    const TensorAdapter<T>* break_point,
                    FixedPointTensor<T, N>* ret) const {

    // e.g., x.shape = {2, 3},
    // break_point.shape = {k, 2, 3} (k: num of break point)
    //       coeff.shape = {k + 1, n, 2, 3} (n: poly power)

    // copy ret
    auto ret_cpy_s0 = tensor_factory()->create_int64_t(ret->share(0)->shape());
    ret->share(0)->copy(ret_cpy_s0.get());
    auto ret_cpy_s1 = tensor_factory()->create_int64_t(ret->share(1)->shape());
    ret->share(1)->copy(ret_cpy_s1.get());
    std::shared_ptr<FixedPointTensor<T, N>> ret_cpy{new FixedPointTensor<T, N>(ret_cpy_s0.get(), ret_cpy_s1.get())};

    std::vector<std::shared_ptr<BooleanTensor<T>>> msb;

    int len_break_point = break_point->shape()[0];
    int len_coeff = coeff->shape()[0];

    //number of temp tensor used
    int temp_total = 4 * len_break_point + 2 +
                     2 * (len_break_point - 1) + 2 + 4 * len_coeff;
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < temp_total; ++i) {
        temp.emplace_back(tensor_factory()->
                          template create<T>(this->shape()));
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

        temp1.emplace_back(
                    std::make_shared<FixedPointTensor<T, N>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
        this->sub(t_break.get(), temp1[i].get());
        msb.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
        msb[i]->bit_extract(sizeof(T) * 8 - 1, temp1[i].get());
    }

    // b[0] = msb[0], b[i + 1] = ~ msb[i] & msb[i + 1]
    std::vector<std::shared_ptr<BooleanTensor<T>>> b;
    b.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
    b[0] = msb[0];

    for (int i = 0; i < len_break_point - 1; ++i) {
        b.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));

        msb[i]->bitwise_not(b[i + 1].get());
        b[i + 1]->bitwise_and(msb[i + 1].get(), b[i + 1].get());
    }

    b.emplace_back(std::make_shared<BooleanTensor<T>>(
                                    temp[temp_index++].get(),
                                    temp[temp_index++].get()));
    msb[len_break_point - 1]->bitwise_not(b[len_break_point].get());

    // ret += b[i].mul(polynomial(coeff[i]))
    std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp_fixed;
    std::vector<std::shared_ptr<FixedPointTensor<T, N>>> temp_fixed1;

    assign_to_tensor(ret_cpy->_share[0], (T) 0);
    assign_to_tensor(ret_cpy->_share[1], (T) 0);

    for (int i = 0; i < len_coeff; ++i) {
        temp_fixed.emplace_back(
                    std::make_shared<FixedPointTensor<T, N>>(
                                                temp[temp_index++].get(),
                                                temp[temp_index++].get()));
        temp_fixed1.emplace_back(
                    std::make_shared<FixedPointTensor<T, N>>(
                                                temp[temp_index++].get(),
                                                temp[temp_index++].get()));
        auto t = tensor_factory()->template create<T>();
        coeff->slice(i, i + 1, t.get());
        auto t_shape = t->shape();
        // remove leading 1
        t_shape.erase(t_shape.begin());
        t->reshape(t_shape);;
        this->polynomial(t.get(), temp_fixed[i].get());
        b[i]->bit_extract(0, b[i].get());
        b[i]->mul(temp_fixed[i].get(), temp_fixed1[i].get());
        ret_cpy->add(temp_fixed1[i].get(), ret_cpy.get());
    }
    ret_cpy->share(0)->copy(ret->mutable_share(0));
    ret_cpy->share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::lt(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 2; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> sub_result =
        std::make_shared<FixedPointTensor<T, N>>(
                                temp[0].get(), temp[1].get());
    this->sub(rhs, sub_result.get());
    ret->bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::leq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    this->gt(rhs, ret);
    auto tensor_one = tensor_factory()->
                            template create<T>(this->shape());

    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::gt(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 2; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> sub_result =
        std::make_shared<FixedPointTensor<T, N>>(
                                    temp[0].get(), temp[1].get());
    this->sub(rhs, sub_result.get());
    sub_result->negative(sub_result.get());
    ret->template bit_extract(sizeof(T) * 8 - 1, sub_result.get());
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::geq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    this->lt(rhs, ret);
    auto tensor_one = tensor_factory()->
                            template create<T>(this->shape());

    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::eq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {

    this->neq(rhs, ret);
    auto tensor_one = tensor_factory()->template create<T>(this->shape());
    assign_to_tensor(tensor_one.get(), (T) 1);
    ret->bitwise_xor(tensor_one.get(), ret);
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::neq(const CTensor<T, N1...>* rhs,
                                BooleanTensor<T>* ret) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 4; i ++) {
        temp.emplace_back(tensor_factory()->
                                template create<T>(this->shape()));
    }
    std::shared_ptr<BooleanTensor<T>> lt =
            std::make_shared<BooleanTensor<T>>(
                                temp[0].get(), temp[1].get());
    std::shared_ptr<BooleanTensor<T>> gt =
            std::make_shared<BooleanTensor<T>>(
                                temp[2].get(), temp[3].get());

    this->lt(rhs, lt.get());
    this->gt(rhs, gt.get());
    lt->bitwise_or(gt.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::reciprocal(const FixedPointTensor<T, N>* op, FixedPointTensor<T, N>* ret,
                                        size_t iter, double x0) {
    auto temp0 = tensor_factory()->template create<T>(ret->shape());
    auto temp1 = tensor_factory()->template create<T>(ret->shape());
    auto temp2 = tensor_factory()->template create<T>(ret->shape());
    auto temp3 = tensor_factory()->template create<T>(ret->shape());
    std::shared_ptr<FixedPointTensor<T, N>> result =
        std::make_shared<FixedPointTensor<T, N>>(temp0.get(), temp1.get());
    std::shared_ptr<FixedPointTensor<T, N>> x_copy =
        std::make_shared<FixedPointTensor<T, N>>(temp2.get(), temp3.get());
    assign_to_tensor(result->mutable_share(0), (T) 0);
    assign_to_tensor(result->mutable_share(1), (T) 0);
    auto tensor_x0 = tensor_factory()->template create<T>(op->shape());
    assign_to_tensor(tensor_x0.get(), (T)(x0 * pow(2, N)));
    tensor_x0->scaling_factor() = N;
    result->add(tensor_x0.get(), result.get());
    auto tensor_2 = tensor_factory()->template create<T>(op->shape());
    tensor_2->scaling_factor() = N;
    assign_to_tensor(tensor_2.get(), (T)(2 << N));
    for (int i = 0; i < iter; ++i) {
        result->share(0)->copy(x_copy->mutable_share(0));
        result->share(1)->copy(x_copy->mutable_share(1));
        auto res_ptr = result.get();
        op->mul(res_ptr, res_ptr);
        result->negative(res_ptr);
        result->add(tensor_2.get(), res_ptr);
        x_copy->mul(res_ptr, res_ptr);
    }
    result->share(0)->copy(ret->mutable_share(0));
    result->share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::inverse_square_root(FixedPointTensor* ret,
                                                 size_t iter,
                                                 double x0) const {
    inverse_square_root(this, ret, iter, x0);
}

// Newton's method, var naming from Quake III Arena: Q_rsqrt
// float threehalfs = 1.5F;
// x2 = number * 0.5F;
// y  = x0; // since 0x5f3759df does not fit fixed-point arithmetic
// y  = y * ( threehalfs - ( x2 * y * y ) ); // iteration of Newton's method
template<typename T, size_t N>
void FixedPointTensor<T, N>::inverse_square_root(const FixedPointTensor* op,
                                                 FixedPointTensor* ret,
                                                 size_t iter,
                                                 double x0) {
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 7; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(op->shape()));
    }
    std::shared_ptr<FixedPointTensor<T, N>> y =
        std::make_shared<FixedPointTensor<T, N>>(temp[0].get(), temp[1].get());
    std::shared_ptr<FixedPointTensor<T, N>> x2 =
        std::make_shared<FixedPointTensor<T, N>>(temp[2].get(), temp[3].get());
    // x2 = 0.5 * op
    truncate(op, x2.get(), 1);

    assign_to_tensor(y->mutable_share(0), (T)(x0 * pow(2, N)));
    assign_to_tensor(y->mutable_share(1), (T)(x0 * pow(2, N)));

    // threehalfs
    temp[4]->scaling_factor() = N;
    assign_to_tensor(temp[4].get(), T(1.5 * pow(2, N)));

    std::shared_ptr<FixedPointTensor<T, N>> y_copy =
        std::make_shared<FixedPointTensor<T, N>>(temp[5].get(), temp[6].get());

    for (int i = 0; i < iter; ++i) {
        y->share(0)->copy(y_copy->mutable_share(0));
        y->share(1)->copy(y_copy->mutable_share(1));
        y->mul(y.get(), y.get());
        y->mul(x2.get(), y.get());
        y->negative(y.get());
        y->add(temp[4].get(), y.get());
        y_copy->mul(y.get(), y.get());
    }
    y->share(0)->copy(ret->mutable_share(0));
    y->share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
template<template<typename U, size_t...> class CTensor,
            size_t... N1>
void FixedPointTensor<T, N>::max(const CTensor<T, N1...>* rhs,
                                 FixedPointTensor* ret,
                                 BooleanTensor<T>* cmp) const {
    // max = lhs + (rhs - lhs) if rhs > lhs else lhs
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    bool output_cmp = cmp != nullptr;
    // if cmp is not null, store cmp results in cmp
    // else, store them in tmp tensors
    for (int i = 0; i < 2 + 2 * (!output_cmp); ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(this->shape()));
    }
    FixedPointTensor<T, N> delta(temp[0].get(), temp[1].get());
    sub(rhs, &delta);
    BooleanTensor<T> sign;
    if (output_cmp) {
        sign = *cmp;
    } else {
        sign = BooleanTensor<T>(temp[2].get(), temp[3].get());
    }
    sign.template bit_extract(sizeof(T) * 8 - 1, &delta);
    delta.negative(&delta);
    sign.mul(&delta, &delta);
    add(&delta, ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::max_pooling(FixedPointTensor* ret,
                                         BooleanTensor<T>* pos) const {
    size_t k = shape()[0];
    std::vector<std::shared_ptr<TensorAdapter<T>>> tmp;
    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(
            tensor_factory()->template create<T>());
    }

    FixedPointTensor now(tmp[0].get(), tmp[1].get());
    BooleanTensor<T> cmp(tmp[2].get(), tmp[3].get());
    auto cmp_ptr = pos ? &cmp : nullptr;

    share(0)->slice(0, 1, tmp[0].get());
    share(1)->slice(0, 1, tmp[1].get());

    tmp[0]->copy(ret->mutable_share(0));
    tmp[1]->copy(ret->mutable_share(1));

    if (pos) {
        pos->share(0)->slice(0, 1, tmp[2].get());
        pos->share(1)->slice(0, 1, tmp[3].get());

        // set init 1, slice_0 is larger than null
        if (party() == 0 || party() == 2) {
            size_t idx = 2 + (party() == 2);
            assign_to_tensor(tmp[idx].get(), T(1));
            assign_to_tensor(tmp[5 - idx].get(), T(0));
        } else {
            assign_to_tensor(tmp[2].get(), T(0));
            assign_to_tensor(tmp[3].get(), T(0));
        }

    }

    for (size_t i = 1; i < k; ++i) {
        share(0)->slice(i, i + 1, tmp[0].get());
        share(1)->slice(i, i + 1, tmp[1].get());

        if (pos) {
            pos->share(0)->slice(i, i + 1, tmp[2].get());
            pos->share(1)->slice(i, i + 1, tmp[3].get());
        }

        ret->max(&now, ret, cmp_ptr);

    }

    if (pos) {
        pos->onehot_from_cmp();
    }

}

} // namespace aby3
