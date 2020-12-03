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
#include <stdexcept>
#include <vector>

#include "gc_bit.h"
#include "core/privc/gc_integer.h"

namespace privc {

const unsigned int taylor_n = 6;

template<size_t N>
int64_t double_to_fix64(double in) {
  return (int64_t) (in * std::pow(2, N));
}

template<size_t N>
double fix64_to_double(int64_t in) {
  return in / std::pow(2, N);
}

inline int64_t factorial(unsigned int i) {
    int64_t ret = 1;
    for (; i > 0; i -= 1) {
        ret *= i;
    }
    return ret;
}

template<size_t N>
class FixedPoint : public privc::IntegerTensor {

public:
    FixedPoint(IntegerTensor &&in) : IntegerTensor(in) {}

    FixedPoint(const IntegerTensor &in) : IntegerTensor(in) {}
    FixedPoint(const std::vector<size_t>& shape) : IntegerTensor(shape) {}
    FixedPoint(const TensorAdapter<int64_t>* input, size_t party) :
                IntegerTensor(input, party) {}

    FixedPoint(double in, std::vector<size_t> shape) : IntegerTensor(get_gc_shape(shape)) {
        int64_t in_ = double_to_fix64<N>(in);
        for (int i = 0; i < _length; i += 1) {
            if (party() == 0 && in_ >> i & 1) {
                auto share_i = (*this)[i];
                auto garbled_delta = tensor_factory()->template create<int64_t>(share_i->shape());
                ot()->garbled_delta(garbled_delta.get());
                garbled_delta->copy(share_i->mutable_share());
            }
        }
    }

    void bitwise_mul(const FixedPoint* rhs, FixedPoint* ret) const {
        PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                          "input numel no match.");
        PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                          "input numel no match.");
        PADDLE_ENFORCE_EQ(size(), rhs->size(),
                          "input size no match.");

        std::vector<size_t> shape = this->shape();

        const unsigned int full_size = size() + N;
        std::vector<size_t> shape_mul = shape;
        shape_mul[0] = full_size;
        IntegerTensor l_(shape_mul);
        IntegerTensor r_(shape_mul);
        IntegerTensor res_(shape_mul);

        for (int i = 0; i < size(); i += 1) {
            (*this)[i]->share()->copy(l_[i]->mutable_share());
            (*rhs)[i]->share()->copy(r_[i]->mutable_share());

        }

        for (int i = 0; (unsigned)i < N; i += 1) {
            (*this)[size() - 1]->share()->copy(l_[size() + i]->mutable_share());
            (*rhs)[size() - 1]->share()->copy(r_[size() + i]->mutable_share());
        }

        mul_full(&res_, &l_, &r_, full_size);

        auto ret_ = tensor_factory()->template create<int64_t>(shape);
        res_.share()->slice(N, full_size, ret_.get());
        ret_->copy(ret->mutable_share());
    }

    void relu(FixedPoint* ret) const {
        PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                          "input numel no match.");

        auto plain_shape = this->shape();
        plain_shape.erase(plain_shape.begin());
        plain_shape.erase(plain_shape.begin());
        FixedPoint zero(0.0, plain_shape);
        auto bit_shape = shape();
        bit_shape.erase(bit_shape.begin());
        BitTensor cmp(bit_shape);
        zero.geq(this, &cmp);
        if_then_else(&cmp, &zero, this, ret);
    }

    void relu_bc(TensorAdapter<int64_t>* ret) {
        PADDLE_ENFORCE_EQ(share()->numel() / shape()[0] / shape()[1],
                    ret->numel(), "input numel no match.");
        auto plain_shape = this->shape();
        plain_shape.erase(plain_shape.begin());
        plain_shape.erase(plain_shape.begin());
        FixedPoint zero(0.0, plain_shape);
        auto bit_shape = shape();
        bit_shape.erase(bit_shape.begin());
        BitTensor cmp(bit_shape);
        zero.geq(this, &cmp);
        if_then_else_bc(&cmp, &zero, this, ret);
    }

    void logistic(FixedPoint* ret) const {
        PADDLE_ENFORCE_EQ(share()->numel(),
                    ret->share()->numel(), "input numel no match.");
        auto gc_shape = this->shape();
        auto bit_shape = gc_shape;
        bit_shape.erase(bit_shape.begin());
        auto plain_shape = bit_shape;
        plain_shape.erase(plain_shape.begin());
        FixedPoint one(1.0, plain_shape);
        FixedPoint half(0.5, plain_shape);
        FixedPoint tmp(gc_shape);
        bitwise_add(&half, &tmp);
        tmp.relu(&tmp);
        BitTensor cmp(bit_shape);
        one.geq(&tmp, &cmp);
        if_then_else(&cmp, &tmp, &one, ret);
    }

};

} // namespace privc

