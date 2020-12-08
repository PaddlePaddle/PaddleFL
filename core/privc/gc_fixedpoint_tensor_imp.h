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

#include "gc_fixedpoint_tensor.h"

#include <algorithm>
#include <vector>

#include "gc_bit.h"
#include "utils.h"
#include "core/common/paddle_tensor.h"

namespace privc {

template<size_t N>
void add_full(GCFixedPointTensor<N> *dest, BitTensor *carry_out,
              const GCFixedPointTensor<N> *op1, const GCFixedPointTensor<N> *op2,
              const BitTensor *carry_in, int size, size_t pos_dest = 0,
              size_t pos_op1 = 0, size_t pos_op2 = 0) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> tmp;
    tmp = tensor_factory()->malloc_tensor<int64_t>(4, bit_shape);
    
    BitTensor carry(tmp[0]);
    BitTensor bxc(tmp[1]);
    BitTensor axc(tmp[2]);
    BitTensor t(tmp[3]);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (carry_in && carry_out) {
            carry_in->share()->copy(carry_out->mutable_share());
        }
        return;
    }
    if (carry_in) {
        carry_in->share()->copy(carry.mutable_share());
    } else {
        carry.set_false();
    }
    // skip AND on last bit if carry_out==NULL
    skip_last = (carry_out == nullptr);
    while (size-- > skip_last) {
        (*op1)[i + pos_op1]->bitwise_xor(&carry, &axc);
        (*op2)[i + pos_op2]->bitwise_xor(&carry, &bxc);
        (*op1)[i + pos_op1]->bitwise_xor(&bxc, (*dest)[i + pos_dest].get());
        axc.bitwise_and(&bxc, &t);
        carry.bitwise_xor(&t, &carry);
        ++i;
    }
    if (carry_out != nullptr) {
        carry.share()->copy(carry_out->mutable_share());
    } else {
        carry.bitwise_xor((*op2)[i + pos_op2].get(), (*dest)[i + pos_dest].get());
        (*dest)[i + pos_dest]->
            bitwise_xor((*op1)[i + pos_op1].get(), (*dest)[i + pos_dest].get());
    }
        
}

template<size_t N>
void sub_full(GCFixedPointTensor<N> *dest, BitTensor *borrow_out,
              const GCFixedPointTensor<N> *op1, const GCFixedPointTensor<N> *op2,
              const BitTensor *borrow_in, int size, int pos_dest = 0,
              int pos_op1 = 0, int pos_op2 = 0) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> tmp;
    tmp = tensor_factory()->malloc_tensor<int64_t>(4, bit_shape);

    BitTensor borrow(tmp[0]);
    BitTensor bxc(tmp[1]);
    BitTensor bxa(tmp[2]);
    BitTensor t(tmp[3]);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (borrow_in && borrow_out) {
            //borrow_out = borrow_in;
            borrow_in->share()->copy(borrow_out->mutable_share());
        }
        return;
    }
    if (borrow_in) {
        // borrow = borrow_in;
        borrow_in->share()->copy(borrow.mutable_share());
    } else {
        borrow.set_false();
    }
    // skip AND on last bit if borrow_out==NULL
    skip_last = (borrow_out == nullptr);
    while (size-- > skip_last) {
        (*op1)[i + pos_op1]->bitwise_xor((*op2)[i + pos_op2].get(), &bxa);
        borrow.bitwise_xor((*op2)[i + pos_op2].get(), &bxc);
        bxa.bitwise_xor(&borrow, (*dest)[i + pos_dest].get());
        bxa.bitwise_and(&bxc, &t);
        borrow.bitwise_xor(&t, &borrow);
        ++i;
    }
    if (borrow_out != nullptr) {
        // borrow_out = borrow;
        borrow.share()->copy(borrow_out->mutable_share());
    } else {
        (*op1)[i + pos_op1]->bitwise_xor((*op2)[i + pos_op2].get(), (*dest)[i + pos_dest].get());
        (*dest)[i + pos_dest]->bitwise_xor(&borrow, (*dest)[i + pos_dest].get());
    }
}

template<size_t N>
void mul_full(GCFixedPointTensor<N> *dest, const GCFixedPointTensor<N> *op1,
              const GCFixedPointTensor<N> *op2, int size) {
    GCFixedPointTensor<N> sum(dest->shape());
    GCFixedPointTensor<N> tmp(dest->shape());

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size - i; ++k) {
            (*op1)[k]->bitwise_and((*op2)[i].get(), tmp[k].get());
        }
        add_full(&sum, nullptr, &sum, &tmp, nullptr, size - i, i, i, 0);
    }
    // calc dest sign
    (*op1)[size - 1]->bitwise_xor((*op2)[size - 1].get(), sum[size - 1].get());
    sum.share()->copy(dest->mutable_share());
}

template<size_t N>
void cond_neg(BitTensor* cond, GCFixedPointTensor<N> *dest,
              const GCFixedPointTensor<N> *src) {
    int i = 0;
    BitTensor c(cond->shape());
    cond->share()->copy(c.mutable_share());
    for (i = 0; i < dest->size() - 1; ++i) {

        (*src)[i]->bitwise_xor(cond, (*dest)[i].get());
        BitTensor t(c.shape());
        (*dest)[i]->bitwise_xor(&c, &t);
        c.bitwise_and((*dest)[i].get(), &c);
 
        t.share()->copy((*dest)[i]->mutable_share());
    }

    c.bitwise_xor(cond, &c);
    c.bitwise_xor((*src)[i].get(), (*dest)[i].get());
}

template<size_t N>
void div_full(GCFixedPointTensor<N> *vquot, GCFixedPointTensor<N> *vrem,
              const GCFixedPointTensor<N> *op1, const GCFixedPointTensor<N> *op2) {

    PADDLE_ENFORCE_EQ(op1->share()->numel(), op2->share()->numel(),
                      "input numel no match");
    auto shape = op1->shape();
    auto size = op1->size();

    GCFixedPointTensor<N> overflow(shape);
    GCFixedPointTensor<N> tmp(shape);
    GCFixedPointTensor<N> rem(shape);
    GCFixedPointTensor<N> quot(shape);

    auto bit_shape = shape;
    bit_shape.erase(bit_shape.begin());
    BitTensor b(bit_shape);
    b.set_false();
    op1->share()->copy(rem.mutable_share());

    for (int i = 1; i < size; ++i) {
        overflow[i - 1]->bitwise_or((*op2)[size - i].get(), overflow[i].get());
    }
    // skip AND on last bit if borrow_out==NULL
    for (int i = size - 1; i >= 0; --i) {
        sub_full(&tmp, &b, &rem, op2, nullptr, size - i, 0, i, 0);
        b.bitwise_or(overflow[i].get(), &b);
        if_then_else(&rem, &rem, &tmp, size - i, &b, i, i, 0);
        b.bitwise_not(quot[i].get());
    }
    if (vrem != nullptr) {
        // vrem = rem
        rem.share()->copy(vrem->mutable_share());
    }
    if (vquot != nullptr) {
        // vquot = quot
        quot.share()->copy(vquot->mutable_share());
    }
}

template<size_t N>
void if_then_else(GCFixedPointTensor<N>* dest, const GCFixedPointTensor<N> *tsrc,
                  const GCFixedPointTensor<N> *fsrc, int size,
                  BitTensor* cond, int pos_dest,
                  int pos_tsrc, int pos_fsrc) {
    int i = 0;
    while (size-- > 0) {
        // dest[i] = cond[i] & (t[i] ^ f[i]) ^ f[i]
        BitTensor x((*tsrc)[i + pos_tsrc]->shape());
        (*tsrc)[i + pos_tsrc]->bitwise_xor((*fsrc)[i + pos_fsrc].get(), &x);
        BitTensor a((*tsrc)[i + pos_tsrc]->shape());
        cond->bitwise_and(&x, &a);
        a.bitwise_xor((*fsrc)[i + pos_fsrc].get(), (*dest)[i + pos_dest].get());
        ++i;
    }
}

template<size_t N>
GCFixedPointTensor<N>::GCFixedPointTensor(const TensorAdapter<int64_t>* input, size_t party_in) {
    // construct gc integer from ac input
    _length = sizeof(int64_t) * 8; // 1 byte = 8 bits

    // gc integer shape = (_length, 2, input.shape)
    auto shape = input->shape();
    auto bit_shape = shape;
    bit_shape.insert(bit_shape.begin(), _length);

    auto gc_shape = bit_shape;
    gc_shape.insert(gc_shape.begin() + 1, _g_block_size_expand);

    auto input_bits = tensor_factory()->template create<u8>(bit_shape);
    // expand input to bit tensor
    to_bits(input, input_bits.get());

    _bits_tensor = tensor_factory()->template create<int64_t>(gc_shape);
    if (party_in == 0) {
        if (party() == 0) {
            auto to_send = tensor_factory()->template create<int64_t>(gc_shape);
            privc_ctx()->template gen_random_private(*to_send);
            to_send->copy(_bits_tensor.get());

            auto mask_val = tensor_factory()->template create<int64_t>(gc_shape);
            auto garbled_delta = tensor_factory()->template create<int64_t>(gc_shape);
            auto zero_block_tensor = tensor_factory()->template create<int64_t>(gc_shape);
            std::for_each(zero_block_tensor->data(),
                          zero_block_tensor->data() + zero_block_tensor->numel(),
                          [](int64_t& a) { a = 0; });
            ot()->garbled_delta(garbled_delta.get());
            if_then_else_plain(input_bits.get(), garbled_delta.get(),
                               zero_block_tensor.get(), mask_val.get());

            to_send->bitwise_xor(mask_val.get(), to_send.get());
            net()->send(next_party(), *to_send);
        } else {
            net()->recv(next_party(), *_bits_tensor);
        }
    } else {
        garbled_share(input_bits.get(), _bits_tensor.get());
    }
}

template<size_t N>
std::shared_ptr<BitTensor> GCFixedPointTensor<N>::operator[](int index) {
    index = std::min(index, size() - 1);
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    auto ret = tensor_factory()->template create<int64_t>(bit_shape);

    _bits_tensor->slice(index, index + 1, ret.get());
    ret->reshape(bit_shape);
    return std::make_shared<BitTensor>(ret);
}

template<size_t N>
const std::shared_ptr<BitTensor> GCFixedPointTensor<N>::operator[](int index) const {
    index = std::min(index, size() - 1);
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    auto ret = tensor_factory()->template create<int64_t>(bit_shape);

    _bits_tensor->slice(index, index + 1, ret.get());
    ret->reshape(bit_shape);
    return std::make_shared<BitTensor>(ret);
}

// Comparisons
template<size_t N>
void GCFixedPointTensor<N>::geq(const GCFixedPointTensor<N> *rhs, BitTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel() / shape()[0],
                      ret->share()->numel(),
                      "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel() / shape()[0],
                      ret->share()->numel(),
                      "input of rhs's numel no match.");

    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }

    GCFixedPointTensor<N> dest(shape());
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    BitTensor borrow_out(bit_shape);

    sub_full(&dest, &borrow_out, this, rhs, nullptr, size());

    (*this)[size() - 1]->bitwise_xor((*rhs)[size() - 1].get(), ret);
    ret->bitwise_xor(&borrow_out, ret);
    ret->bitwise_not(ret);
}

template<size_t N>
void GCFixedPointTensor<N>::equal(const GCFixedPointTensor<N>* rhs, BitTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel() / shape()[0], ret->share()->numel(),
                      "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel() / shape()[0],
                      ret->share()->numel(),
                      "input of rhs's numel no match.");

    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());

    BitTensor res(bit_shape);
    BitTensor tmp(bit_shape);

    res.bitwise_not(&res);
    for (int i = 0; i < size(); ++i) {

        (*this)[i]->bitwise_xor((*rhs)[i].get(), &tmp);
        tmp.bitwise_not(&tmp);
        res.bitwise_and(&tmp, &res);
    }
    res.share()->copy(ret->mutable_share());
}

template<size_t N>
void GCFixedPointTensor<N>::is_zero(BitTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel() / shape()[0],
                     ret->share()->numel(), "input numel no match.");
    for (int i = 0; i < size(); ++i) {
        ret->bitwise_or((*this)[i].get(), ret);
    }

}

template<size_t N>
void GCFixedPointTensor<N>::abs(GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");

    GCFixedPointTensor<N> res(shape());
    for (int i = 0; i < size(); ++i) {
        (*this)[size() - 1]->share()->copy(res[i]->mutable_share());
    }
    bitwise_add(&res, ret);
    ret->bitwise_xor(&res, ret);
}

template<size_t N>
void GCFixedPointTensor<N>::bitwise_xor(const GCFixedPointTensor<N>* rhs,
                                GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                      "input of rhs's numel no match with return.");

    share()->bitwise_xor(rhs->share(), ret->mutable_share());
}

template<size_t N>
void GCFixedPointTensor<N>::bitwise_add(const GCFixedPointTensor<N>* rhs,
                                GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                      "input of rhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(size(), rhs->size(), "input size no match.");

    add_full(ret, nullptr, this, rhs, nullptr, size());
}

template<size_t N>
void GCFixedPointTensor<N>::bitwise_sub(const GCFixedPointTensor<N>* rhs,
                                GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                      "input of rhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(size(), rhs->size(), "input size no match.");

    sub_full(ret, nullptr, this, rhs, nullptr, size());
}

template<size_t N>
void GCFixedPointTensor<N>::bitwise_neg(GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");

    GCFixedPointTensor<N> zero(shape());
    sub_full(ret, nullptr, &zero, this, nullptr, size());
}

template<size_t N>
void GCFixedPointTensor<N>::if_then_else(BitTensor* cond, const GCFixedPointTensor<N>* t_int,
                              const GCFixedPointTensor<N>* f_int,
                              GCFixedPointTensor<N>* ret) {
    PADDLE_ENFORCE_EQ(cond->share()->numel() * sizeof(int64_t) * 8,
                      ret->share()->numel(),
                      "input of condition's numel no match with return.");
    PADDLE_ENFORCE_EQ(t_int->share()->numel(),
                      ret->share()->numel(),
                      "input of true val's numel no match with return.");
    PADDLE_ENFORCE_EQ(f_int->share()->numel(),
                      ret->share()->numel(),
                      "input of false val's numel no match with return.");

    GCFixedPointTensor<N> res(t_int->shape());
    privc::if_then_else(&res, t_int, f_int, res.size(), cond);

    res.share()->copy(ret->mutable_share());
}

template<size_t N>
void GCFixedPointTensor<N>::if_then_else_bc(BitTensor* cond,
                                const GCFixedPointTensor<N>* t_int,
                                const GCFixedPointTensor<N>* f_int,
                                TensorAdapter<int64_t>* ret) {
    PADDLE_ENFORCE_EQ(cond->share()->numel() / cond->shape()[0],
                      ret->numel(),
                      "input of condition's numel no match with return.");
    PADDLE_ENFORCE_EQ(t_int->share()->numel() / t_int->shape()[0] / t_int->shape()[1],
                      ret->numel(),
                      "input of true val's numel no match with return.");
    PADDLE_ENFORCE_EQ(f_int->share()->numel() / f_int->shape()[0] / f_int->shape()[1],
                      ret->numel(),
                      "input of false val's numel no match with return.");

    // convert gc input to bc 
    auto lsb_cond = tensor_factory()->create<u8>(ret->shape());
    auto lsb_t_int = tensor_factory()->template create<int64_t>(ret->shape());
    auto lsb_f_int = tensor_factory()->template create<int64_t>(ret->shape());
    block_lsb(cond->share(), lsb_cond.get());
    t_int->lsb(lsb_t_int.get());
    f_int->lsb(lsb_f_int.get());

    // cal bc_mux
    bc_mux(lsb_cond.get(), lsb_t_int.get(), lsb_f_int.get(), ret);
}

template<size_t N>
void GCFixedPointTensor<N>::bitwise_mul(const GCFixedPointTensor<N>* rhs, GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                        "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                        "input of rhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(size(), rhs->size(),
                        "input size no match.");

    std::vector<size_t> shape = this->shape();

    const unsigned int full_size = size() + N;
    std::vector<size_t> shape_mul = shape;
    shape_mul[0] = full_size;
    GCFixedPointTensor<N> l_(shape_mul);
    GCFixedPointTensor<N> r_(shape_mul);
    GCFixedPointTensor<N> res_(shape_mul);

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

template<size_t N>
void GCFixedPointTensor<N>::bitwise_div(const GCFixedPointTensor<N>* rhs, GCFixedPointTensor<N>* ret) const {

    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                    "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                    "input of rhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(size(), rhs->size(), "input size no match.");

    GCFixedPointTensor<N> i1(shape());
    GCFixedPointTensor<N> i2(shape());
    abs(&i1);
    rhs->abs(&i2);

    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    BitTensor sign(bit_shape);
    (*this)[size() - 1]->bitwise_xor((*rhs)[size() - 1].get(), &sign);

    const size_t full_size = size() + N;

    auto full_shape = shape();
    full_shape[0] = full_size;

    GCFixedPointTensor<N> l_(full_shape);
    GCFixedPointTensor<N> r_(full_shape);
    GCFixedPointTensor<N> res_(full_shape);

    std::copy(i1.share()->data(), i1.share()->data() + i1.share()->numel(),
                l_.mutable_share()->data() + N * l_.share()->numel() / full_size);
    std::copy(i2.share()->data(), i2.share()->data() + i2.share()->numel(),
                r_.mutable_share()->data());

    div_full<N>(&res_, nullptr, &l_, &r_);

    BitTensor q_sign(bit_shape);
    res_[size() - 1]->share()->copy(q_sign.mutable_share());

    GCFixedPointTensor<N> nan(shape());
    for (int i = 0; i < size() - 1; ++i) {
        q_sign.share()->copy(nan[i]->mutable_share());
    }
    q_sign.bitwise_not(nan[size() - 1].get());

    privc::if_then_else(&res_, &nan, &res_, size(), &q_sign);

    cond_neg(&sign, &res_, &res_);

    BitTensor tmp(bit_shape);
    sign.bitwise_and(&q_sign, &tmp);
    res_[0]->bitwise_xor(&tmp, res_[0].get());

    std::copy(res_.share()->data(), res_.share()->data() + ret->share()->numel(),
                ret->mutable_share()->data());
}

template<size_t N>
void GCFixedPointTensor<N>::relu(GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                        "input numel no match.");

    auto plain_shape = this->shape();
    plain_shape.erase(plain_shape.begin());
    plain_shape.erase(plain_shape.begin());
    GCFixedPointTensor<N> zero(0.0, plain_shape);
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    BitTensor cmp(bit_shape);
    zero.geq(this, &cmp);
    if_then_else(&cmp, &zero, this, ret);
}

template<size_t N>
void GCFixedPointTensor<N>::relu_bc(TensorAdapter<int64_t>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel() / shape()[0] / shape()[1],
                ret->numel(), "input numel no match.");
    auto plain_shape = this->shape();
    plain_shape.erase(plain_shape.begin());
    plain_shape.erase(plain_shape.begin());
    GCFixedPointTensor<N> zero(0.0, plain_shape);
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    BitTensor cmp(bit_shape);
    zero.geq(this, &cmp);
    if_then_else_bc(&cmp, &zero, this, ret);
}

template<size_t N>
void GCFixedPointTensor<N>::logistic(GCFixedPointTensor<N>* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(),
                ret->share()->numel(), "input numel no match.");
    auto gc_shape = this->shape();
    auto bit_shape = gc_shape;
    bit_shape.erase(bit_shape.begin());
    auto plain_shape = bit_shape;
    plain_shape.erase(plain_shape.begin());
    GCFixedPointTensor<N> one(1.0, plain_shape);
    GCFixedPointTensor<N> half(0.5, plain_shape);
    GCFixedPointTensor<N> tmp(gc_shape);
    bitwise_add(&half, &tmp);
    tmp.relu(&tmp);
    BitTensor cmp(bit_shape);
    one.geq(&tmp, &cmp);
    if_then_else(&cmp, &tmp, &one, ret);
}

inline void get_row_element(int row, const TensorBlock* share, TensorBlock* ret) {
    auto shape = share->shape();
    auto gc_element_size = sizeof(int64_t) * _g_block_size_expand * 8;
    
    auto num_row = shape[2];
    auto num_col = shape[3];
    PADDLE_ENFORCE_GT(num_row, row, "input row large than total row.");

    std::copy(share->data() + row * num_col * gc_element_size,
              share->data() + (row + 1) * num_col * gc_element_size,
              ret->data());
}

inline void get_element_from_vector(int col,
                TensorBlock* share_v, TensorBlock* ret) {
    auto shape = share_v->shape();
    auto gc_element_size = sizeof(int64_t) * _g_block_size_expand * 8;
    
    auto num_col = shape[0];
    PADDLE_ENFORCE_GT(num_col, col, "input col large than total col.");
    PADDLE_ENFORCE_EQ(ret->numel(),
            gc_element_size, "input numel no match.");
    std::copy(share_v->data() + col * gc_element_size,
              share_v->data() + (col + 1) * gc_element_size,
              ret->data());
}

inline block* get_bit_element(int row, int col,
                TensorBlock* bit_tensor) {
    auto shape = bit_tensor->shape();
    
    auto num_col = shape[2];
    PADDLE_ENFORCE_GT(num_col, col, "input col large than total col.");

    block* ret_ptr = reinterpret_cast<block*>(bit_tensor->data()) + row * num_col + col;
    return ret_ptr;
}

template<size_t N>
void GCFixedPointTensor<N>::argmax_one_hot(
                const GCFixedPointTensor<N>* op,
                GCFixedPointTensor<N>* ret) {
    PADDLE_ENFORCE_EQ(ret->shape()[0], 1, "1 size (bit) is enough for ret");
    auto gc_shape = op->shape();
    auto val_shape = gc_shape;
    val_shape.erase(val_shape.begin());
    val_shape.erase(val_shape.begin());
    auto one_bit_shape = get_block_shape({1});

    auto shape_row{val_shape[1]};

    BitTensor max_one_hot(get_block_shape(val_shape));

    auto true_ = tensor_factory()->template create<u8>({1});
    *(true_->data()) = 1;

    // enumerate matrix column
    for (int i = 0; i < val_shape[0]; ++i) {
        // assign first cmp result to true
        block* cmp_i_0 = get_bit_element(i, 0, max_one_hot.mutable_share());
        BitTensor bit_true(true_.get(), 0);
        *cmp_i_0 = *reinterpret_cast<const block*>(bit_true.share()->data());

        // enumerate each element in each column
        auto row_i = tensor_factory()->template create<int64_t>(shape_row);
        get_row_element(i, op->share(), row_i.get());

        GCFixedPointTensor<N> max(get_gc_shape({1}));
        get_element_from_vector(0, row_i.get(), max.mutable_share());
        for(int j = 1; j < val_shape[1]; ++j) {
            GCFixedPointTensor<N> op_i_j(get_gc_shape({1}));
            get_element_from_vector(j, row_i.get(), op_i_j.mutable_share());

            BitTensor cmp(one_bit_shape);
            op_i_j.geq(&max, &cmp);
            block* cmp_i_j = get_bit_element(i, j, max_one_hot.mutable_share());
            *cmp_i_j = *reinterpret_cast<const block*>(cmp.share()->data());
            if_then_else(&cmp, &op_i_j, &max, &max);
        }
    }

    // find max index
    auto false_ = tensor_factory()->template create<u8>({1});
    *(false_->data()) = 0;

    for (int i = 0; i < val_shape[0]; ++i) {
        BitTensor has_found(false_.get(), 0);
        for (int j = val_shape[1] - 1; j >= 0; --j) {
            // found = has_found || bit[j]
            block* bit_share_i_j = get_bit_element(i, j, max_one_hot.mutable_share());
            BitTensor bit_i_j(one_bit_shape);
            *reinterpret_cast<block*>(bit_i_j.mutable_share()->data()) = *bit_share_i_j;

            BitTensor found(one_bit_shape);
            has_found.bitwise_xor(&bit_i_j, &found);
            // bit[j] = !has_found & bit[j]
            BitTensor tmp(one_bit_shape);
            has_found.bitwise_not(&tmp);
            tmp.bitwise_and(&bit_i_j, &bit_i_j);
            *bit_share_i_j = *reinterpret_cast<const block*>(bit_i_j.share()->data());
            // has_found = found
            *reinterpret_cast<block*>(has_found.mutable_share()->data()) =
                        *reinterpret_cast<const block*>(found.share()->data());
        }
    }
    // BitTensor to GCFixedPointTensor<N>
    max_one_hot.share()->copy(ret->mutable_share());
}

void to_ac_num(const TensorAdapter<int64_t>* input,
               TensorAdapter<int64_t>* ret) {
    // convert boolean share to arithmetic share
    PADDLE_ENFORCE_EQ(input->numel(), ret->numel(), "input numel no match.");
    const size_t word_width = sizeof(int64_t) * 8; // 8 bit for 1 byte

    auto shape = input->shape();

    auto gc_shape = get_gc_shape(shape, word_width);
    auto block_shape = get_block_shape(shape);
    auto buffer_shape = gc_shape;
    buffer_shape.erase(buffer_shape.begin() + 1);
    auto ret_ = tensor_factory()->template create<int64_t>(shape);
    std::for_each(ret_->data(), ret_->data() + ret_->numel(),
                  [](int64_t& a) { a = 0; });

    if (party() == 0) {
        auto s1_buffer = tensor_factory()->template create<int64_t>(buffer_shape);
        auto ot_mask = tensor_factory()->template create<int64_t>(gc_shape);
        net()->recv(next_party(), *ot_mask);
        for (size_t idx = 0; idx < word_width; ++idx) {
            // sender
            // with ot input (s0, s1)
            // s0 = (1 - x[i])* 2^i - r_i, s1 = x[i]*2^i - r_i
            auto round_ot_mask = tensor_factory()->template create<int64_t>(block_shape);
            ot_mask->slice(idx, idx + 1, round_ot_mask.get());
            round_ot_mask->reshape(block_shape);

            auto q = tensor_factory()->template create<int64_t>(block_shape);
            ot()->ot_sender().get_ot_instance(q.get());

            auto base_ot_choice = tensor_factory()->template create<int64_t>(block_shape);
            auto tmp = tensor_factory()->template create<int64_t>(block_shape);
            ot()->base_ot_choice(base_ot_choice.get());
            round_ot_mask->bitwise_and(base_ot_choice.get(), tmp.get());
            q->bitwise_xor(tmp.get(), q.get());

            auto s_first = tensor_factory()->template create<int64_t>(block_shape);
            auto s_second = tensor_factory()->template create<int64_t>(block_shape);
            q->bitwise_xor(base_ot_choice.get(), tmp.get());
            std::pair<TensorBlock*, TensorBlock*> x_pair({q.get(), tmp.get()});
            std::pair<TensorBlock*, TensorBlock*> s_pair({s_first.get(), s_second.get()});
            common::hash_blocks(x_pair, s_pair);

            auto s0 = tensor_factory()->template create<int64_t>(shape);
            block_to_int64(s_first.get(), s0.get());
            
            auto bit_mask = tensor_factory()->template create<int64_t>(shape);
            std::for_each(bit_mask->data(), bit_mask->data() + bit_mask->numel(),
                            [&idx](int64_t& a) { a = ((int64_t)1 << idx); });
            auto ai = tensor_factory()->template create<int64_t>(shape);
            input->bitwise_and(bit_mask.get(), ai.get());
            auto r = tensor_factory()->template create<int64_t>(shape);
            ai->sub(s0.get(), r.get());

            auto s1 = tensor_factory()->template create<int64_t>(shape);
            block_to_int64(s_second.get(), s1.get());

            ai->bitwise_xor(bit_mask.get(), ai.get());
            ai->sub(r.get(), ai.get());
            s1->bitwise_xor(ai.get(), s1.get());

            auto s1_buffer_s = tensor_factory()->template create<int64_t>(shape);
            s1_buffer->slice(idx, idx + 1, s1_buffer_s.get());

            s1->copy(s1_buffer_s.get());
            // ret_ =  sum_i{r_i}
            ret_->add(r.get(), ret_.get());
        }
        net()->send(next_party(), *s1_buffer);

    } else {
        // as ot recver
        // with choice bit x[i]
        auto ot_masks = tensor_factory()->template create<int64_t>(gc_shape);
        auto t0_buffer = tensor_factory()->template create<int64_t>(gc_shape);
        auto ot_msg = tensor_factory()->template create<int64_t>(buffer_shape);
        auto& ot_ext_recver = ot()->ot_receiver();
        gen_ot_masks(ot_ext_recver, input, ot_masks.get(), t0_buffer.get());

        net()->send(next_party(), *ot_masks);

        net()->recv(next_party(), *ot_msg);

        for (u64 idx = 0; idx < word_width; idx += 1) {
            auto round_ot_msg = tensor_factory()->template create<int64_t>(shape);
            ot_msg->slice(idx, idx + 1, round_ot_msg.get());
            round_ot_msg->reshape(shape);

            auto t0_buffer_s = tensor_factory()->template create<int64_t>(block_shape);
            t0_buffer->slice(idx, idx + 1, t0_buffer_s.get());

            auto t0_hash = tensor_factory()->template create<int64_t>(block_shape);
            common::hash_block(t0_buffer_s.get(), t0_hash.get());

            auto key = tensor_factory()->template create<int64_t>(shape);
            block_to_int64(t0_hash.get(), key.get());

            auto tmp = tensor_factory()->template create<int64_t>(shape);
            round_ot_msg->bitwise_xor(key.get(), tmp.get());
            auto cond = tensor_factory()->create<int64_t>(shape);
            std::transform(input->data(), input->data() + input->numel(),
                            cond->data(), [&idx] (int64_t a) {
                                return ((a >> idx) & 1);
                            });
            if_then_else_plain(false, cond.get(), tmp.get(), key.get(), tmp.get());
            // ret_ = sum_i{x[i] * 2^i} - sum_i{r_i}
            ret_->add(tmp.get(), ret_.get());
        }
    }
    ret_->copy(ret);
}

void bc_mux(const TensorAdapter<u8>* choice,
            const TensorAdapter<int64_t>* val_t,
            const TensorAdapter<int64_t>* val_f,
            TensorAdapter<int64_t>* ret) {
    PADDLE_ENFORCE_EQ(choice->numel(), ret->numel(),
            "input of choice's numel no match with return.");
    PADDLE_ENFORCE_EQ(val_t->numel(), ret->numel(),
            "input of true val's numel no match with return.");
    PADDLE_ENFORCE_EQ(val_f->numel(), ret->numel(),
            "input of false val's numel no match with return.");

    auto send_ot = [](const TensorAdapter<int64_t>* diff,
                          const TensorBlock* round_ot_mask,
                          TensorAdapter<int64_t>* send_buffer,
                          TensorAdapter<int64_t>* ret) {
        // bad naming from ot extention
        // send_buffer = s1 = diff ^ ot_instance0 ^ ot_instance1
        // ret = s0 = ot_instance0
        auto q = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        ot()->ot_sender().get_ot_instance(q.get());

        auto base_ot_choice = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        ot()->base_ot_choice(base_ot_choice.get());
        auto tmp = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        round_ot_mask->bitwise_and(base_ot_choice.get(), tmp.get());
        q->bitwise_xor(tmp.get(), q.get());

        auto s_first = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        auto s_second = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        q->bitwise_xor(base_ot_choice.get(), tmp.get());
        std::pair<TensorBlock*, TensorBlock*> x_pair({q.get(), tmp.get()});
        std::pair<TensorBlock*, TensorBlock*> s_pair({s_first.get(), s_second.get()});
        common::hash_blocks(x_pair, s_pair);

        auto s0 = tensor_factory()->template create<int64_t>(ret->shape());
        block_to_int64(s_first.get(), s0.get());

        auto msg1 = tensor_factory()->template create<int64_t>(s0->shape());
        diff->bitwise_xor(s0.get(), msg1.get());

        auto s1 = tensor_factory()->template create<int64_t>(send_buffer->shape());
        block_to_int64(s_second.get(), s1.get());

        s1->bitwise_xor(msg1.get(), s1.get());

        s1->copy(send_buffer);

        s0->copy(ret);
    };

    auto send_ot_mask = [](const TensorAdapter<u8>* choice, TensorBlock* send_buffer,
                           TensorAdapter<int64_t>* ret) {
        // ret = H(ot_instance0)
        auto block_shape = send_buffer->shape();
        auto ot_ins0 = tensor_factory()->template create<int64_t>(block_shape);
        auto ot_ins1 = tensor_factory()->template create<int64_t>(block_shape);
        ot()->ot_receiver().get_ot_instance(ot_ins0.get(), ot_ins1.get());

        auto choice_ = tensor_factory()->template create<int64_t>(block_shape);
        block* choice_ptr = reinterpret_cast<block*>(choice_->data());
        std::transform(choice->data(), choice->data() + choice->numel(),
                       choice_ptr, [](u8 val) {
                           return val ? common::OneBlock : common::ZeroBlock;
                        });

        const auto& t0 = ot_ins0;
        // ot_mask = choice ^ ot_instance0 ^ ot_instance1
        auto ot_mask = tensor_factory()->template create<int64_t>(block_shape);
        choice_->bitwise_xor(ot_ins0.get(), ot_mask.get());
        ot_mask->bitwise_xor(ot_ins1.get(), ot_mask.get());

        ot_mask->copy(send_buffer);

        auto t0_hash = tensor_factory()->template create<int64_t>(block_shape);
        common::hash_block(t0.get(), t0_hash.get());
        block_to_int64(t0_hash.get(), ret);
    };

    auto recv_ot = [](const TensorAdapter<u8>* choice, const TensorAdapter<int64_t>* key,
                      const TensorAdapter<int64_t>* round_ot_msg,
                      TensorAdapter<int64_t>* ret) {

        auto tmp = tensor_factory()->template create<int64_t>(key->shape());
        round_ot_msg->bitwise_xor(key, tmp.get());
        if_then_else_plain(false, choice, tmp.get(), key, ret);
    };

    auto shape = ret->shape();
    auto block_shape = get_block_shape(shape);
    auto delta = tensor_factory()->template create<int64_t>(shape);
    auto msg = tensor_factory()->template create<int64_t>(shape);
    auto key = tensor_factory()->template create<int64_t>(shape);

    auto send_buffer = tensor_factory()->template create<int64_t>(shape);
    auto send_buffer1 = tensor_factory()->template create<int64_t>(block_shape);
    auto recv_buffer = tensor_factory()->template create<int64_t>(block_shape);
    auto recv_buffer1 = tensor_factory()->template create<int64_t>(shape);

    auto ret_ = tensor_factory()->template create<int64_t>(shape);


    if (party() == 0) {
        net()->recv(next_party(), *recv_buffer);
        const auto& round_ot_mask = recv_buffer;
        auto tmp = tensor_factory()->template create<int64_t>(shape);
        val_t->bitwise_xor(val_f, tmp.get());
        send_ot(tmp.get(), round_ot_mask.get(), send_buffer.get(), delta.get());
        net()->send(next_party(), *send_buffer);

        send_ot_mask(choice, send_buffer1.get(), key.get());

        net()->send(next_party(), *send_buffer1);
        net()->recv(next_party(), *recv_buffer1);

        const auto& round_ot_msg = recv_buffer1;
        recv_ot(choice, key.get(), round_ot_msg.get(), msg.get());
    } else {

        send_ot_mask(choice, send_buffer1.get(), key.get());

        net()->send(next_party(), *send_buffer1);
        net()->recv(next_party(), *recv_buffer1);

        const auto& round_ot_msg = recv_buffer1;
        recv_ot(choice, key.get(), round_ot_msg.get(), msg.get());

        net()->recv(next_party(), *recv_buffer);

        const auto& round_ot_mask = recv_buffer;
        auto tmp = tensor_factory()->template create<int64_t>(shape);
        val_t->bitwise_xor(val_f, tmp.get());
        send_ot(tmp.get(), round_ot_mask.get(), send_buffer.get(), delta.get());

        net()->send(next_party(), *send_buffer);
    }

    // val_f[i] ^ delta[i] ^ msg[i] ^ choice[i] * (val_t[i] ^ val_f[i])

    val_t->bitwise_xor(val_f, ret_.get());

    std::transform(choice->data(), choice->data() + choice->numel(),
                    ret_->data(), ret_->data(), [](u8 a, int64_t b) {
                        return (a * b);
                    });
    auto tmp = tensor_factory()->template create<int64_t>(shape);
    val_f->bitwise_xor(delta.get(), tmp.get());
    tmp->bitwise_xor(msg.get(), tmp.get());
    tmp->bitwise_xor(ret_.get(), ret);
}

}; // namespace smc
