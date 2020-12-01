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

#include "bit.h"
#include "core/privc/integer.h"

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

    
/*
    FixedPoint(double input, size_t party)
        : IntegerTensor(double_to_fix64<N>(input), party) {
    }

    double reconstruct() const {
        return fix64_to_double<N>(IntegerTensor::reconstruct());
    }

    FixedPoint decimal() const {
        FixedPoint res = abs();
        for (int i = N; i < res.size(); i += 1) {
            res[i]._share = ZeroBlock;
        }
        cond_neg(_bits[size() - 1], res.bits(), res.cbits(), size());
        return res;
    }
*/
    /*FixedPoint operator*(const FixedPoint &rhs) const {
        if (size() != rhs.size()) {
            throw std::logic_error("op len not match");
        }
        FixedPoint res(*this);

        const unsigned int full_size = size() + N;

        std::vector<Bit> l_vec;
        std::vector<Bit> r_vec;
        std::vector<Bit> res_vec(full_size, Bit());

        for (int i = 0; i < size(); i += 1) {
            l_vec.emplace_back(_bits[i]);
            r_vec.emplace_back(rhs[i]);
        }

        for (int i = 0; (unsigned)i < N; i += 1) {
            l_vec.emplace_back(_bits[size() - 1]);
            r_vec.emplace_back(rhs[size() - 1]);
        }

        mul_full(res_vec.data(), l_vec.data(), r_vec.data(), full_size);

        for (int i = 0; i < size(); i += 1) {
            res[i] = std::move(res_vec[i + N]);
        }
        return res;
    }
*/
    void bitwise_mul(const FixedPoint* rhs, FixedPoint* ret) const {
        if (size() != rhs->size()) {
            throw std::logic_error("op len not match");
        }
        std::vector<size_t> shape = this->shape();
        //FixedPoint res(shape);

        const unsigned int full_size = size() + N;
        std::vector<size_t> shape_mul = shape;
        shape_mul[0] = full_size;
        IntegerTensor l_(shape_mul);
        IntegerTensor r_(shape_mul);
        IntegerTensor res_(shape_mul);

        for (int i = 0; i < size(); i += 1) {
            //l_vec.emplace_back(_bits[i]);
            //r_vec.emplace_back(rhs[i]);
            (*this)[i]->share()->copy(l_[i]->mutable_share());
            (*rhs)[i]->share()->copy(r_[i]->mutable_share());

        }

        for (int i = 0; (unsigned)i < N; i += 1) {
            //l_vec.emplace_back(_bits[size() - 1]);
            //r_vec.emplace_back(rhs[size() - 1]);
            (*this)[size() - 1]->share()->copy(l_[size() + i]->mutable_share());
            (*rhs)[size() - 1]->share()->copy(r_[size() + i]->mutable_share());
        }

        mul_full(&res_, &l_, &r_, full_size);

        //for (int i = 0; i < size(); i += 1) {
        //    res[i] = std::move(res_vec[i + N]);
        //}
        //return res;
        auto ret_ = tensor_factory()->template create<int64_t>(shape);
        res_.share()->slice(N, full_size, ret_.get());
        ret_->copy(ret->mutable_share());
    }
/*
    FixedPoint operator/(const FixedPoint &rhs) const {
        if (size() != rhs.size()) {
            throw std::logic_error("op len not match");
        }
        FixedPoint res(*this);
        FixedPoint i1 = abs();
        FixedPoint i2 = rhs.abs();
        Bit sign = _bits[size() - 1] ^ rhs[size() - 1];

        const unsigned int full_size = size() + N;

        std::vector<Bit> l_vec(N, Bit());
        std::vector<Bit> r_vec;
        std::vector<Bit> res_vec(full_size);

        for (int i = 0; i < size(); i += 1) {
            l_vec.emplace_back(std::move(i1[i]));
            r_vec.emplace_back(std::move(i2[i]));
        }

        for (int i = 0; (unsigned)i < N; i += 1) {
            r_vec.emplace_back(Bit());
        }

        div_full(res_vec.data(), nullptr, l_vec.data(), r_vec.data(),
                 full_size);

        Bit q_sign = res_vec[size() - 1];
        std::vector<Bit> nan(size(), q_sign);
        nan[size() - 1] = ~q_sign;

        privc::if_then_else(res_vec.data(), nan.data(), res_vec.data(), size(), q_sign);

        cond_neg(sign, res_vec.data(), res_vec.data(), full_size);

        res_vec[0] = res_vec[0] ^ (q_sign & sign);

        for (int i = 0; i < size(); i += 1) {
            res[i] = std::move(res_vec[i]);
        }
        return res;
    }

    FixedPoint exp_int() const {
        // e^22 > 2^31 - 1, 22 = 0x16
        // e^-22 = 2.79 * 10 ^ -10 sufficiently precise
        return exp(_bits[size() - 1], abs().bits() + N, 5);
    }

    FixedPoint exp_gc() const {
        auto exp_int_ = exp_int();
        auto x = decimal() * FixedPoint(0.5);
        auto x_n = FixedPoint(1.0);

        std::vector<FixedPoint> var;
        for (int i = 0; (unsigned)i <= taylor_n; i += 1) {
            var.emplace_back(x_n);
            x_n = x_n * x;
        }

        auto exp_dec = var[0];
        for (unsigned int i = 1; i <= taylor_n; i += 1) {
            exp_dec = exp_dec + var[i] * FixedPoint(1.0 / factorial(i));
        }
        return exp_int_ * exp_dec * exp_dec;
    }*/

    //FixedPoint relu() const {
    //    FixedPoint zero(0.0);
    //    return if_then_else(zero.geq(*this), zero, *this);
    //}
    void relu(FixedPoint* ret) const {
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
/*    int64_t relu_bc() const {
        FixedPoint zero(0.0);
        return if_then_else_bc(zero.geq(*this), zero, *this);
    }
*/
    void relu_bc(TensorAdapter<int64_t>* ret) {
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

/*    FixedPoint logistic() const {
        FixedPoint one(1.0);
        FixedPoint half(0.5);
        FixedPoint t_option = FixedPoint(operator+(half)).relu();
        return if_then_else(one.geq(t_option), t_option, one);
    }*/

    void logistic(FixedPoint* ret) const {
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
/*
    static std::vector<FixedPoint> softmax(std::vector<FixedPoint> &&in) {
        if (in.size() == 0) {
            throw std::logic_error("zero input vector size");
        }
        FixedPoint sum(0.0);
        for (auto &x: in) {
            x = x.relu();
            sum = sum + x;
        }
        auto sum_zero = sum.is_zero();
        FixedPoint avg(1.0 / in.size());
        for (auto &x: in) {
            x = if_then_else(sum_zero, avg, x / sum);
        }
        return in;
    }

private:
    static FixedPoint exp(const Bit &neg, const Bit *in, int size) {
        FixedPoint res(1.0);

        FixedPoint base = IntegerTensor::if_then_else(neg, FixedPoint(1.0 / M_E),
                                                FixedPoint(M_E));

        FixedPoint one = res;

        for (int i = size - 1; i >= 0; i -= 1) {
            FixedPoint round = IntegerTensor::if_then_else(in[i], base, one);
            res = res * round;
            if (i) {
                res = res * res;
            }
        }
        return res;
    }*/
};

template<size_t N>
using Fix64gc = FixedPoint<N>;

} // namespace privc

