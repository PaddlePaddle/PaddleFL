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

#include "integer.h"

#include <algorithm>
#include <vector>

#include "bit.h"
#include "common_utils.h"

namespace privc {

/*void add_full(Bit *dest, Bit *carry_out, const Bit *op1, const Bit *op2,
              const Bit *carry_in, int size) {
    Bit carry;
    Bit bxc;
    Bit axc;
    Bit t;

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (carry_in && carry_out) {
            *carry_out = *carry_in;
        }
        return;
    }
    if (carry_in) {
        carry = *carry_in;
    } else {
        // carry = false;
        carry = Bit();
    }
    // skip AND on last bit if carry_out==NULL
    skip_last = (carry_out == nullptr);
    while (size-- > skip_last) {
        axc = op1[i] ^ carry;
        bxc = op2[i] ^ carry;
        dest[i] = op1[i] ^ bxc;
        t = axc & bxc;
        carry = carry ^ t;
        ++i;
    }
    if (carry_out != nullptr) {
        *carry_out = carry;
    }
    else
        dest[i] = carry ^ op2[i] ^ op1[i];
}*/

void add_full(IntegerTensor *dest, BitTensor *carry_out,
              const IntegerTensor *op1, const IntegerTensor *op2,
              const BitTensor *carry_in, int size, size_t pos_dest = 0,
              size_t pos_op1 = 0, size_t pos_op2 = 0) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> tmp;
    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(tensor_factory()->template create<int64_t>(bit_shape));
    }
    BitTensor carry(tmp[0]);
    BitTensor bxc(tmp[1]);
    BitTensor axc(tmp[2]);
    BitTensor t(tmp[3]);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (carry_in && carry_out) {
            //*carry_out = *carry_in;
            carry_in->share()->copy(carry_out->mutable_share());
        }
        return;
    }
    if (carry_in) {
        //carry = *carry_in;
        carry_in->share()->copy(carry.mutable_share());
    } else {
        // carry = false;
        //carry = BitTensor();
        carry.set_false();
    }
    // skip AND on last bit if carry_out==NULL
    skip_last = (carry_out == nullptr);
    while (size-- > skip_last) {
        //axc = op1[i] ^ carry;
        (*op1)[i + pos_op1]->bitwise_xor(&carry, &axc);
        //bxc = op2[i] ^ carry;
        (*op2)[i + pos_op2]->bitwise_xor(&carry, &bxc);
        //dest[i] = op1[i] ^ bxc;
        (*op1)[i + pos_op1]->bitwise_xor(&bxc, (*dest)[i + pos_dest].get());
        //t = axc & bxc;
        axc.bitwise_and(&bxc, &t);
        //carry = carry ^ t;
        carry.bitwise_xor(&t, &carry);
        ++i;
    }
    if (carry_out != nullptr) {
        //*carry_out = carry;
        carry.share()->copy(carry_out->mutable_share());
    } else {
        //dest[i] = carry ^ op2[i] ^ op1[i];
        carry.bitwise_xor((*op2)[i + pos_op2].get(), (*dest)[i + pos_dest].get());
        (*dest)[i + pos_dest]->bitwise_xor((*op1)[i + pos_op1].get(), (*dest)[i + pos_dest].get());
    }
        
}
/*
void sub_full(Bit *dest, Bit *borrow_out, const Bit *op1, const Bit *op2,
              const Bit *borrow_in, int size) {
    Bit borrow;
    Bit bxc;
    Bit bxa;
    Bit t;

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (borrow_in && borrow_out) {
            *borrow_out = *borrow_in;
        }
        return;
    }
    if (borrow_in) {
        borrow = *borrow_in;
    } else {
        // borrow = false;
        borrow = Bit();
    }
    // skip AND on last bit if borrow_out==NULL
    skip_last = (borrow_out == nullptr);
    while (size-- > skip_last) {
        bxa = op1[i] ^ op2[i];
        bxc = borrow ^ op2[i];
        dest[i] = bxa ^ borrow;
        t = bxa & bxc;
        borrow = borrow ^ t;
        ++i;
    }
    if (borrow_out != nullptr) {
        *borrow_out = borrow;
    } else {
        dest[i] = op1[i] ^ op2[i] ^ borrow;
    }
}*/

void sub_full(IntegerTensor *dest, BitTensor *borrow_out, const IntegerTensor *op1, const IntegerTensor *op2,
              const BitTensor *borrow_in, int size) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> tmp;
    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(tensor_factory()->template create<int64_t>(bit_shape));
    }
    BitTensor borrow(tmp[0]);
    BitTensor bxc(tmp[1]);
    BitTensor bxa(tmp[2]);
    BitTensor t(tmp[3]);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (borrow_in && borrow_out) {
            *borrow_out = *borrow_in;
        }
        return;
    }
    if (borrow_in) {
        borrow = *borrow_in;
    } else {
        // borrow = false;
        //borrow = Bit();
        borrow.set_false();
    }
    // skip AND on last bit if borrow_out==NULL
    skip_last = (borrow_out == nullptr);
    while (size-- > skip_last) {
        //bxa = op1[i] ^ op2[i];
        (*op1)[i]->bitwise_xor((*op2)[i].get(), &bxa);
        //bxc = borrow ^ op2[i];
        borrow.bitwise_xor((*op2)[i].get(), &bxc);
        //dest[i] = bxa ^ borrow;
        bxa.bitwise_xor(&borrow, (*dest)[i].get());
        //t = bxa & bxc;
        bxa.bitwise_and(&bxc, &t);
        //borrow = borrow ^ t;
        borrow.bitwise_xor(&t, &borrow);
        ++i;
    }
    if (borrow_out != nullptr) {
        *borrow_out = borrow;
    } else {
        //dest[i] = op1[i] ^ op2[i] ^ borrow;
        (*op1)[i]->bitwise_xor((*op2)[i].get(), (*dest)[i].get());
        (*dest)[i]->bitwise_xor(&borrow, (*dest)[i].get());
    }
}
/*
void mul_full(Bit *dest, const Bit *op1, const Bit *op2, int size) {

    std::vector<Bit> sum(size, Bit());
    std::vector<Bit> tmp(size, Bit());

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size - i; ++k) {
            tmp[k] = op1[k] & op2[i];
        }
        add_full(sum.data() + i, nullptr, sum.data() + i, tmp.data(), nullptr, size - i);
    }
    memcpy(dest, sum.data(), sizeof(Bit) * size);
}*/

void mul_full(IntegerTensor *dest, const IntegerTensor *op1, const IntegerTensor *op2, int size) {

    //std::vector<Bit> sum(size, Bit());
    //std::vector<Bit> tmp(size, Bit());
    IntegerTensor sum(dest->shape());
    IntegerTensor tmp(dest->shape());

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size - i; ++k) {
            //tmp[k] = op1[k] & op2[i];
            (*op1)[k]->bitwise_and((*op2)[i].get(), tmp[k].get());
        }
        //add_full(sum.data() + i, nullptr, sum.data() + i, tmp.data(), nullptr, size - i);
        add_full(&sum, nullptr, &sum, &tmp, nullptr, size - i, i, i, 0);
    }
    //memcpy(dest, sum.data(), sizeof(Bit) * size);
    sum.share()->copy(dest->mutable_share());
}


/*void if_then_else(Bit *dest, const Bit *tsrc, const Bit *fsrc, int size,
                  Bit cond) {
    int i = 0;
    while (size-- > 0) {
        auto x = tsrc[i] ^ fsrc[i];
        auto a = cond & x;
        dest[i] = a ^ fsrc[i];
        ++i;
    }
}*/

void if_then_else(IntegerTensor* dest, const IntegerTensor *tsrc, const IntegerTensor *fsrc, int size,
                  BitTensor* cond) {
    int i = 0;
    while (size-- > 0) {
        BitTensor x((*tsrc)[i]->shape());
        //auto x = tsrc[i] ^ fsrc[i];
        (*tsrc)[i]->bitwise_xor((*fsrc)[i].get(), &x);
        BitTensor a((*tsrc)[i]->shape());
        //auto a = cond & x;
        cond->bitwise_and(&x, &a);
        //dest[i] = a ^ fsrc[i];
        a.bitwise_xor((*fsrc)[i].get(), (*dest)[i].get());
        ++i;
    }
}

/*void cond_neg(Bit cond, Bit *dest, const Bit *src, int size) {
    int i = 0;
    Bit c = cond;
    for (i = 0; i < size - 1; ++i) {
        dest[i] = src[i] ^ cond;
        Bit t = dest[i] ^ c;
        c = c & dest[i];
        dest[i] = t;
    }
    dest[i] = cond ^ c ^ src[i];
}*/
/*
void div_full(Bit *vquot, Bit *vrem, const Bit *op1, const Bit *op2, int size) {
    std::vector<Bit> overflow(size, Bit());
    std::vector<Bit> tmp(size);
    std::vector<Bit> rem(size);
    std::vector<Bit> quot(size);
    Bit b;
    memcpy(rem.data(), op1, size * sizeof(Bit));
    for (int i = 1; i < size; ++i) {
        overflow[i] = overflow[i - 1] | op2[size - i];
    }
    // skip AND on last bit if borrow_out==NULL
    for (int i = size - 1; i >= 0; --i) {
        sub_full(tmp.data(), &b, rem.data() + i, op2, nullptr, size - i);
        b = b | overflow[i];
        if_then_else(rem.data() + i, rem.data() + i, tmp.data(), size - i, b);
        quot[i] = ~b;
    }
    if (vrem != nullptr) {
        memcpy(vrem, rem.data(), size * sizeof(Bit));
    }
    if (vquot != nullptr) {
        memcpy(vquot, quot.data(), size * sizeof(Bit));
    }
}*/
/*
IntegerTensor::IntegerTensor(int64_t input, size_t party_in)
    : _length(sizeof(input) * 8), _bits(_length) {
    if (party_in == 0) {
        if (party() == 0) {
            std::vector<block> send_buffer;
            for (u64 idx = 0; idx < sizeof(input) * 8; idx += 1) {

                block to_send = privc_ctx()->gen_random_private<block>();
                _bits[idx]._share = to_send;

                to_send ^=
                    (input >> idx) & 1 ? ot()->garbled_delta() : common::ZeroBlock;
                send_buffer.emplace_back(to_send);
                //_ctx->send_to_buffer(to_send);
            }
            net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(block));

        } else {
            std::vector<block> recv_buffer;
            recv_buffer.resize(_length);
            net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(block));
            for (int idx = 0; idx < _length; idx += 1) {
                _bits[idx]._share = recv_buffer[idx];
            }
        }
    } else {
        auto bits = garbled_share(input);
        for (int idx = 0; idx < _length; idx += 1) {
            _bits[idx]._share = bits[idx];
        }
    }
}
*/

IntegerTensor::IntegerTensor(const TensorAdapter<int64_t>* input, size_t party_in) {
    _length = sizeof(int64_t) * 8;
    auto shape = input->shape();
    auto bit_shape = shape;
    bit_shape.insert(bit_shape.begin(), _length);
    //size_t block_size_expand = sizeof(block) / sizeof(int64_t);
    auto gc_shape = bit_shape;
    gc_shape.insert(gc_shape.begin() + 1, _g_block_size_expand);

    auto input_bits = tensor_factory()->template create<u8>(bit_shape);
    to_bits(input, input_bits.get());
    //std::cout << "input: "<<input<<"\n";
    //std::cout << "input_bits: "<<input_bits.get()<<"\n";
    _bits_tensor = tensor_factory()->template create<int64_t>(gc_shape);
    if (party_in == 0) {
        if (party() == 0) {
            //std::vector<block> send_buffer;
            //for (u64 idx = 0; idx < sizeof(input) * 8; idx += 1) {

                //block to_send = privc_ctx()->gen_random_private<block>();
                //_bits[idx]._share = to_send;

                //to_send ^=
                //    (input >> idx) & 1 ? ot()->garbled_delta() : common::ZeroBlock;
                //send_buffer.emplace_back(to_send);
                //_ctx->send_to_buffer(to_send);
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
            if_then_else_plain(input_bits.get(), garbled_delta.get(), zero_block_tensor.get(), mask_val.get());

            to_send->bitwise_xor(mask_val.get(), to_send.get());
            //}
            net()->send(next_party(), *to_send);

        } else {
            //std::vector<block> recv_buffer;
            //recv_buffer.resize(_length);
            //net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(block));
            //for (int idx = 0; idx < _length; idx += 1) {
            //    _bits[idx]._share = recv_buffer[idx];
            //}
            //TODO: recv without shape
            net()->recv(next_party(), *_bits_tensor);
        }
    } else {
        //auto bits = garbled_share(input);
        //for (int idx = 0; idx < _length; idx += 1) {
        //    _bits[idx]._share = bits[idx];
        //}
        garbled_share(input_bits.get(), _bits_tensor.get());
    }
}

/*
std::vector<IntegerTensor> IntegerTensor::vector(const std::vector<int64_t>& input,
                                     size_t party_in) {
    const size_t length = sizeof(int64_t) * 8;
    std::vector<IntegerTensor> ret;
    if (party_in == 0) {
        if (party() == 0) {
            std::vector<block> send_buffer;
            for (size_t i = 0; i < input.size(); ++i) {
                IntegerTensor int_i;
                int_i._length = length;
                int_i._bits.resize(length);
                for (u64 idx = 0; idx < length; idx += 1) {

                    block to_send = privc_ctx()->gen_random_private<block>();
                    int_i._bits[idx]._share = to_send;

                    to_send ^=
                        (input[i] >> idx) & 1 ? ot()->garbled_delta() : common::ZeroBlock;
                    //ctx->send_to_buffer(to_send);
                    send_buffer.emplace_back(to_send);
                }
                ret.emplace_back(int_i);

            }
            net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(block));
            //ctx->flush_buffer();

        } else {

            std::vector<block> recv_buffer;
            recv_buffer.resize(input.size() * length);
            net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(block));
            size_t buffer_idx = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                IntegerTensor int_i;
                int_i._length = length;
                int_i._bits.resize(length);
                for (size_t idx = 0; idx < length; idx += 1) {
                    int_i._bits[idx]._share = recv_buffer[buffer_idx];
                    buffer_idx++;
                }
                ret.emplace_back(int_i);
            }
        }
    } else {
        auto bits = garbled_share(input);
        for (size_t i = 0; i < input.size(); ++i) {
            IntegerTensor int_i;
            int_i._length = length;
            int_i._bits.resize(length);
            for (size_t idx = 0; idx < length; idx += 1) {
                int_i._bits[idx]._share = bits[i * length + idx];
            }
            ret.emplace_back(int_i);
        }
    }
    return ret;
}
*/
std::shared_ptr<BitTensor> IntegerTensor::operator[](int index) {
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    auto ret = tensor_factory()->template create<int64_t>(bit_shape);

    _bits_tensor->slice(index, index + 1, ret.get());
    ret->reshape(bit_shape);
    return std::make_shared<BitTensor>(ret);
}

const std::shared_ptr<BitTensor> IntegerTensor::operator[](int index) const {
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    auto ret = tensor_factory()->template create<int64_t>(bit_shape);

    _bits_tensor->slice(index, index + 1, ret.get());
    ret->reshape(bit_shape);
    return std::make_shared<BitTensor>(ret);
}

// Comparisons
void IntegerTensor::geq(const IntegerTensor *rhs, BitTensor* ret) const {
    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }
    //IntegerTensor tmp = (*this) - rhs;

    //std::vector<Bit> dest(size());
    //Bit borrow_out;
    IntegerTensor dest(shape());
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    BitTensor borrow_out(bit_shape);

    sub_full(&dest, &borrow_out, this, rhs, nullptr, size());
    //return ~(borrow_out ^ _bits[size() - 1] ^ rhs._bits[size() - 1]);
    (*this)[size() - 1]->bitwise_xor((*rhs)[size() - 1].get(), ret);
    ret->bitwise_xor(&borrow_out, ret);
    ret->bitwise_not(ret);
}

void IntegerTensor::equal(const IntegerTensor* rhs, BitTensor* ret) const {
    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());

    BitTensor res(bit_shape);
    BitTensor tmp(bit_shape);
    //res = ~res;
    res.bitwise_not(&res);
    for (int i = 0; i < size(); ++i) {
        //res = res & ~(_bits[i] ^ rhs._bits[i]);
        (*this)[i]->bitwise_xor((*rhs)[i].get(), &tmp);
        tmp.bitwise_not(&tmp);
        res.bitwise_and(&tmp, &res);
    }
    res.share()->copy(ret->mutable_share());
}

void IntegerTensor::is_zero(BitTensor* ret) const {

    for (int i = 0; i < size(); ++i) {
        //res = res | _bits[i];
        ret->bitwise_or((*this)[i].get(), ret);
    }
    //return ~res;
}

void IntegerTensor::abs(IntegerTensor* ret) const {
    IntegerTensor res(shape());
    for (int i = 0; i < size(); ++i) {
        (*this)[size() - 1]->share()->copy(res[i]->mutable_share());
    }
    //return ((*this) + res) ^ res;
    bitwise_add(&res, ret);
    ret->bitwise_xor(&res, ret);
}

void IntegerTensor::bitwise_xor(const IntegerTensor* rhs, IntegerTensor* ret) const {
    share()->bitwise_xor(rhs->share(), ret->mutable_share());
}

// Arithmethics
/*IntegerTensor IntegerTensor::operator+(const IntegerTensor &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    IntegerTensor res(*this);
    add_full(res.bits(), nullptr, cbits(), rhs.cbits(), nullptr, size());
    return res;
}*/

void IntegerTensor::bitwise_add(const IntegerTensor* rhs, IntegerTensor* ret) const {
    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }
    add_full(ret, nullptr, this, rhs, nullptr, size());
}
/*
IntegerTensor IntegerTensor::operator-(const IntegerTensor &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    IntegerTensor res(*this);
    sub_full(res.bits(), nullptr, cbits(), rhs.cbits(), nullptr, size());
    return res;
}*/

void IntegerTensor::bitwise_sub(const IntegerTensor* rhs, IntegerTensor* ret) const {
    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }
    sub_full(ret, nullptr, this, rhs, nullptr, size());
}

/*IntegerTensor IntegerTensor::operator*(const IntegerTensor &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    IntegerTensor res(*this);
    mul_full(res.bits(), cbits(), rhs.cbits(), size());
    return res;
}*/

void IntegerTensor::bitwise_mul(const IntegerTensor* rhs, IntegerTensor* ret) const {
    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }
    //share()->copy(ret->mutable_share());
    mul_full(ret, this, rhs, size());
}
/*
IntegerTensor IntegerTensor::operator/(const IntegerTensor &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    IntegerTensor res(*this);
    IntegerTensor i1 = abs();
    IntegerTensor i2 = rhs.abs();
    Bit sign = _bits[size() - 1] ^ rhs[size() - 1];
    div_full(res.bits(), nullptr, i1.cbits(), i2.cbits(), size());
    Bit q_sign = res[size() - 1];
    std::vector<Bit> nan(size(), q_sign);
    nan[size() - 1] = ~q_sign;

    privc::if_then_else(res.bits(), nan.data(), res.cbits(), size(), q_sign);

    cond_neg(sign, res.bits(), res.cbits(), size());
    res[0] = res[0] ^ (sign & q_sign);
    return res;
}*/

/*IntegerTensor IntegerTensor::operator-() const {
    std::vector<Bit> zero(size());
    IntegerTensor res(*this);
    sub_full(res.bits(), nullptr, zero.data(), cbits(), nullptr, size());
    return res;
}*/

void IntegerTensor::bitwise_neg(IntegerTensor* ret) const {
    IntegerTensor zero(shape());
    sub_full(ret, nullptr, &zero, this, nullptr, size());
}

/*IntegerTensor IntegerTensor::if_then_else(Bit cond, const IntegerTensor &t_int,
                              const IntegerTensor &f_int) {
    IntegerTensor res(t_int);
    privc::if_then_else(res.bits(), t_int.cbits(), f_int.cbits(), res.size(), cond);

    return res;
}*/

void IntegerTensor::if_then_else(BitTensor* cond, const IntegerTensor* t_int,
                              const IntegerTensor* f_int,
                              IntegerTensor* ret) {
    IntegerTensor res(t_int->shape());
    privc::if_then_else(&res, t_int, f_int, res.size(), cond);
   
    //return res;
    res.share()->copy(ret->mutable_share());
}
/*
int64_t IntegerTensor::if_then_else_bc(Bit cond, const IntegerTensor &t_int,
                                const IntegerTensor &f_int) {
    return bc_mux(block_lsb(cond._share), t_int.lsb(), f_int.lsb());
}*/

void IntegerTensor::if_then_else_bc(BitTensor* cond, const IntegerTensor* t_int,
                                const IntegerTensor* f_int, TensorAdapter<int64_t>* ret) {
    auto lsb_cond = tensor_factory()->create<u8>(ret->shape());
    auto lsb_t_int = tensor_factory()->template create<int64_t>(ret->shape());
    auto lsb_f_int = tensor_factory()->template create<int64_t>(ret->shape());
    block_lsb(cond->share(), lsb_cond.get());
    t_int->lsb(lsb_t_int.get());
    f_int->lsb(lsb_f_int.get());
    bc_mux(lsb_cond.get(), lsb_t_int.get(), lsb_f_int.get(), ret);
}
/*
int64_t IntegerTensor::argmax(const std::vector<IntegerTensor>& op, size_t party_in) {
    size_t size = op.size();
    if (size <= 0) {
        throw std::logic_error("op len error");
    }
    if (size == 1) {
        return 0;
    }
    std::vector<Bit> cmp_bit;
    cmp_bit.resize(size - 1);

    IntegerTensor max = op[0];
    for(int i = 1; i < size; ++i) {
        auto cmp = op[i].geq(max);
        max = if_then_else(cmp, op[i], max);
        cmp_bit.emplace_back(cmp);
    }

    // find max index
    std::vector<bool> cmp_plain = privc::reconstruct(cmp_bit, party_in);
    auto pos = std::find(cmp_plain.rbegin(), cmp_plain.rend(), true);
    if (pos != cmp_plain.rend()) {
        return std::distance(pos, cmp_plain.rend());
    } else {
        return 0;
    }
}
*/

/*
void IntegerTensor::argmax_one_hot(
                const IntegerTensor* op,
                IntegerTensor* ret) {
    std::vector<int64_t> ret;
    size_t size = op.size();
    if (size <= 0) {
        throw std::logic_error("op len error");
    }
    if (size == 1) {
        ret.emplace_back(Bit(1,0).lsb());
        return ret;
    }
    std::vector<Bit> cmp_bit;

    IntegerTensor max = op[0];
    for(int i = 1; i < size; ++i) {
        auto cmp = op[i].geq(max);
        max = if_then_else(cmp, op[i], max);
        cmp_bit.emplace_back(cmp);
    }

    // find max index
    Bit found(false, 0);
    std::vector<Bit> res_b;
    res_b.resize(size);
    res_b[size - 1] = cmp_bit[size - 2];

    for (int i = size - 2; i > 0; --i) {
        found = cmp_bit[i] || found;
        res_b[i] = !found & cmp_bit[i - 1];
    }
    found = cmp_bit[0] || found;
    res_b[0] = !found;

    // convert Bit to IntegerTensor
    ret.resize(res_b.size());
    for( int i = 0; i < ret.size(); ++i) {
      ret[i] = (int64_t)res_b[i].lsb();
    }
    return ret;
}*/


/*
std::vector<int64_t> to_ac_num_internal(const int64_t* input, size_t size) {
    const size_t word_width = sizeof(int64_t) * 8; // 8 bit for 1 byte

    std::vector<int64_t> ret(size);

    if (party() == 0) {

        std::vector<int64_t> s1_buffer;
        std::vector<block> ot_mask;
        ot_mask.resize(size * word_width);
        net()->recv(next_party(), ot_mask.data(), ot_mask.size() * sizeof(block));
        size_t ot_mask_idx = 0;
        for (size_t i = 0; i < size; ++i) {
            for (size_t idx = 0; idx < word_width; ++idx) {

                const block& round_ot_mask = ot_mask.at(ot_mask_idx);
                //_io->recv_data(&round_ot_mask, sizeof(block));

                block q = ot()->ot_sender().get_ot_instance();

                q ^= (round_ot_mask & ot()->base_ot_choice());

                auto s = common::hash_blocks({q, q ^ ot()->base_ot_choice()});
                int64_t  s0 = *reinterpret_cast<int64_t *>(&s.first);

                int64_t bit_mask = (int64_t )1 << idx;
                int64_t ai = input[i] & bit_mask;
                auto r = ai - s0;

                int64_t  s1 = *reinterpret_cast<int64_t *>(&s.second);

                s1 ^= (ai ^ bit_mask) - r;

                s1_buffer.emplace_back(s1);

                ret[i] += r;
                ot_mask_idx++;
            }
        }
        net()->send(next_party(), s1_buffer.data(), sizeof(int64_t) * s1_buffer.size());

    } else { // as ot recver

        std::vector<block> ot_masks;
        std::vector<block> t0_buffer;
        std::vector<int64_t> ot_msg;
        auto& ot_ext_recver = ot()->ot_receiver();
        gen_ot_masks(ot_ext_recver, input, size, ot_masks, t0_buffer);
        net()->send(next_party(), ot_masks.data(), sizeof(block) * ot_masks.size());
        ot_msg.resize(size * word_width);
        size_t ot_msg_idx = 0;
        net()->recv(next_party(), ot_msg.data(), ot_msg.size() * sizeof(int64_t));

        for (size_t i = 0; i < size; ++i) {
            for (u64 idx = 0; idx < word_width; idx += 1) {
                const int64_t& round_ot_msg = ot_msg.at(ot_msg_idx);
                //_io->recv_data(&ot_msg, sizeof(ot_msg));

                auto t0_hash = common::hash_block(t0_buffer[i * word_width + idx]);

                int64_t key = *reinterpret_cast<int64_t *>(&t0_hash);
                ret[i] += (input[i] >> idx) & 1 ? round_ot_msg ^ key : key;
                ot_msg_idx++;
            }
        }

    }

    return ret;
}

void to_ac_num(const TensorAdapter<int64_t>* input,
               TensorAdapter<int64_t>* ret) {
    std::vector<int64_t> ret_ = to_ac_num_internal(input->data(), input->numel());
    std::copy(ret_.data(), ret_.data() + ret_.size(), ret->data());
}*/

void to_ac_num(const TensorAdapter<int64_t>* input,
               TensorAdapter<int64_t>* ret) {
    const size_t word_width = sizeof(int64_t) * 8; // 8 bit for 1 byte

    //std::vector<int64_t> ret(size);
    auto shape = input->shape();
    auto gc_shape = get_gc_shape(shape, word_width);
    auto block_shape = get_block_shape(shape);
    auto buffer_shape = gc_shape;
    buffer_shape.erase(buffer_shape.begin() + 1);
    auto ret_ = tensor_factory()->template create<int64_t>(shape);

    if (party() == 0) {

        //std::vector<int64_t> s1_buffer;
        //std::vector<block> ot_mask;
        //ot_mask.resize(size * word_width);
        auto s1_buffer = tensor_factory()->template create<int64_t>(buffer_shape);
        auto ot_mask = tensor_factory()->template create<int64_t>(gc_shape);
        //net()->recv(next_party(), ot_mask.data(), ot_mask.size() * sizeof(block));
        net()->recv(next_party(), *ot_mask);
        //size_t ot_mask_idx = 0;
        //for (size_t i = 0; i < size; ++i) {
            for (size_t idx = 0; idx < word_width; ++idx) {

                //const block& round_ot_mask = ot_mask.at(ot_mask_idx);
                auto round_ot_mask = tensor_factory()->template create<int64_t>(block_shape);
                ot_mask->slice(idx, idx + 1, round_ot_mask.get());
                round_ot_mask->reshape(block_shape);

                //block q = ot()->ot_sender().get_ot_instance();
                auto q = tensor_factory()->template create<int64_t>(block_shape);
                ot()->ot_sender().get_ot_instance(q.get());

                //q ^= (round_ot_mask & ot()->base_ot_choice());
                auto base_ot_choice = tensor_factory()->template create<int64_t>(block_shape);
                auto tmp = tensor_factory()->template create<int64_t>(block_shape);
                ot()->base_ot_choice(base_ot_choice.get());
                round_ot_mask->bitwise_and(base_ot_choice.get(), tmp.get());
                q->bitwise_xor(tmp.get(), q.get());

                //auto s = common::hash_blocks({q, q ^ ot()->base_ot_choice()});
                auto s_first = tensor_factory()->template create<int64_t>(block_shape);
                auto s_second = tensor_factory()->template create<int64_t>(block_shape);
                q->bitwise_xor(base_ot_choice.get(), tmp.get());
                std::pair<TensorBlock*, TensorBlock*> x_pair({q.get(), tmp.get()});
                std::pair<TensorBlock*, TensorBlock*> s_pair({s_first.get(), s_second.get()});
                common::hash_blocks(x_pair, s_pair);

                //int64_t  s0 = *reinterpret_cast<int64_t *>(&s.first);
                auto s0 = tensor_factory()->template create<int64_t>(shape);
                block_to_int64(s_first.get(), s0.get());

                //int64_t bit_mask = (int64_t )1 << idx;
                //int64_t ai = input[i] & bit_mask;
                //auto r = ai - s0;
                
                auto bit_mask = tensor_factory()->template create<int64_t>(shape);
                std::for_each(bit_mask->data(), bit_mask->data() + bit_mask->numel(),
                              [&idx](int64_t& a) { a = ((int64_t)1 << idx); });
                auto ai = tensor_factory()->template create<int64_t>(shape);
                input->bitwise_and(bit_mask.get(), ai.get());
                auto r = tensor_factory()->template create<int64_t>(shape);
                ai->sub(s0.get(), r.get());

                //int64_t  s1 = *reinterpret_cast<int64_t *>(&s.second);
                auto s1 = tensor_factory()->template create<int64_t>(shape);
                block_to_int64(s_second.get(), s1.get());

                //s1 ^= (ai ^ bit_mask) - r;
                ai->bitwise_xor(bit_mask.get(), ai.get());
                ai->sub(r.get(), ai.get());
                s1->bitwise_xor(ai.get(), s1.get());


                //s1_buffer.emplace_back(s1);
                auto s1_buffer_s = tensor_factory()->template create<int64_t>(shape);
                s1_buffer->slice(idx, idx + 1, s1_buffer_s.get());
                s1_buffer_s->reshape(shape);
                s1->copy(s1_buffer_s.get());

                //ret[i] += r;
                ret_->add(r.get(), ret_.get());
                //ot_mask_idx++;
            }
        //}
        //net()->send(next_party(), s1_buffer.data(), sizeof(int64_t) * s1_buffer.size());
        net()->send(next_party(), *s1_buffer);

    } else { // as ot recver

        //std::vector<block> ot_masks;
        //std::vector<block> t0_buffer;
        //std::vector<int64_t> ot_msg;
        auto ot_masks = tensor_factory()->template create<int64_t>(gc_shape);
        auto t0_buffer = tensor_factory()->template create<int64_t>(gc_shape);
        auto ot_msg = tensor_factory()->template create<int64_t>(buffer_shape);
        auto& ot_ext_recver = ot()->ot_receiver();
        gen_ot_masks(ot_ext_recver, input, ot_masks.get(), t0_buffer.get());
        //net()->send(next_party(), ot_masks.data(), sizeof(block) * ot_masks.size());
        net()->send(next_party(), *ot_masks);
        //ot_msg.resize(size * word_width);
        //size_t ot_msg_idx = 0;
        //net()->recv(next_party(), ot_msg.data(), ot_msg.size() * sizeof(int64_t));
        net()->recv(next_party(), *ot_msg);

        //for (size_t i = 0; i < size; ++i) {
            for (u64 idx = 0; idx < word_width; idx += 1) {
                //const int64_t& round_ot_msg = ot_msg.at(ot_msg_idx);
                auto round_ot_msg = tensor_factory()->template create<int64_t>(shape);
                ot_msg->slice(idx, idx + 1, round_ot_msg.get());
                round_ot_msg->reshape(shape);

                //auto t0_hash = common::hash_block(t0_buffer[i * word_width + idx]);
                auto t0_buffer_s = tensor_factory()->template create<int64_t>(block_shape);
                t0_buffer->slice(idx, idx + 1, t0_buffer_s.get());
                t0_buffer_s->reshape(block_shape);
                auto t0_hash = tensor_factory()->template create<int64_t>(block_shape);
                common::hash_block(t0_buffer_s.get(), t0_hash.get());

                //int64_t key = *reinterpret_cast<int64_t *>(&t0_hash);
                auto key = tensor_factory()->template create<int64_t>(shape);
                block_to_int64(t0_hash.get(), key.get());
                //ret[i] += (input[i] >> idx) & 1 ? round_ot_msg ^ key : key;
                auto tmp = tensor_factory()->template create<int64_t>(shape);
                round_ot_msg->bitwise_xor(key.get(), tmp.get());
                auto cond = tensor_factory()->create<u8>(shape);
                std::transform(input->data(), input->data() + input->numel(),
                               cond->data(), [&idx] (int64_t a) -> u8 {
                                   return (u8) ((a >> idx) & (u8) 1);
                                });
                if_then_else_plain(cond.get(), tmp.get(), key.get(), tmp.get());
                ret_->add(tmp.get(), ret_.get());
                //ot_msg_idx++;
            }
        //}

    }
    ret_->copy(ret);
    //return ret;
}

/*int64_t to_ac_num(int64_t val) {
    return to_ac_num_internal(&val, 1)[0];
}

std::vector<int64_t> to_ac_num(const std::vector<int64_t>& input) {
    return to_ac_num_internal(input.data(), input.size());
}*/
/*
std::vector<int64_t> bc_mux_internal(const uint8_t* choice,
                                                     const int64_t* val_t,
                                                     const int64_t* val_f,
                                                     size_t size) {
    auto send_ot = [](int64_t diff,
                          const block& round_ot_mask,
                          std::vector<int64_t>& send_buffer) {
        //block round_ot_mask = recv_val<block>();

        // bad naming from ot extention
        block q = ot()->ot_sender().get_ot_instance();

        q ^= (round_ot_mask & ot()->base_ot_choice());

        auto s = common::hash_blocks({q, q ^ ot()->base_ot_choice()});
        int64_t  s0 = *reinterpret_cast<int64_t *>(&s.first);

        int64_t msg1 = diff ^ s0;

        int64_t  s1 = *reinterpret_cast<int64_t *>(&s.second);

        s1 ^= msg1;

        //send_to_buffer(s1);
        send_buffer.emplace_back(s1);

        return s0;
    };

    auto send_ot_mask = [](bool choice, std::vector<block>& send_buffer) {
        auto ot_instance = ot()->ot_receiver().get_ot_instance();
        block choice_ = choice ? common::OneBlock : common::ZeroBlock;

        const auto& t0 = ot_instance[0];
        auto ot_mask = choice_ ^ ot_instance[0] ^ ot_instance[1];

        //send_to_buffer(ot_mask);
        send_buffer.emplace_back(ot_mask);

        auto t0_hash = common::hash_block(t0);
        int64_t key = *reinterpret_cast<int64_t *>(&t0_hash);

        return key;
    };

    auto recv_ot = [](bool choice, int64_t key, const int64_t& round_ot_msg) {
        //int64_t ot_msg = recv_val<int64_t>();

        return choice ? round_ot_msg ^ key : key;

    };

    std::vector<int64_t> delta;
    std::vector<int64_t> msg;
    std::vector<int64_t> key;

    std::vector<int64_t> send_buffer;
    std::vector<block> send_buffer1;
    std::vector<block> recv_buffer;
    std::vector<int64_t> recv_buffer1;
    recv_buffer.resize(size);
    recv_buffer1.resize(size);

    if (party() == 0) {
        net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(block));
        for (size_t i = 0; i < size; ++i) {
            const block& round_ot_mask = recv_buffer.at(i);
            delta.emplace_back(send_ot(val_t[i] ^ val_f[i], round_ot_mask, send_buffer));
        }
        //flush_buffer();
        net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(int64_t));
        for (size_t i = 0; i < size; ++i) {
            key.emplace_back(send_ot_mask(choice[i], send_buffer1));
        }
        //flush_buffer();
        net()->send(next_party(), send_buffer1.data(), send_buffer1.size() * sizeof(block));
        net()->recv(next_party(), recv_buffer1.data(), recv_buffer1.size() * sizeof(int64_t));
        for (size_t i = 0; i < size; ++i) {
            const int64_t& round_ot_msg = recv_buffer1.at(i);
            msg.emplace_back(recv_ot(choice[i], key[i], round_ot_msg));
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            key.emplace_back(send_ot_mask(choice[i], send_buffer1));
        }
        //flush_buffer();
        net()->send(next_party(), send_buffer1.data(), send_buffer1.size() * sizeof(block));
        net()->recv(next_party(), recv_buffer1.data(), recv_buffer1.size() * sizeof(int64_t));
        for (size_t i = 0; i < size; ++i) {
            const int64_t& round_ot_msg = recv_buffer1.at(i);
            msg.emplace_back(recv_ot(choice[i], key[i], round_ot_msg));
        }
        net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(block));
        for (size_t i = 0; i < size; ++i) {
            const block& round_ot_mask = recv_buffer.at(i);
            delta.emplace_back(send_ot(val_t[i] ^ val_f[i], round_ot_mask, send_buffer));
        }
        //flush_buffer();
        net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(int64_t));
    }

    std::vector<int64_t> ret;
    for (size_t i = 0; i < size; ++i) {
        ret.emplace_back(val_f[i] ^ delta[i] ^ msg[i] ^ choice[i] * (val_t[i] ^ val_f[i]));
    }

    return ret;
}
*/
void bc_mux(const TensorAdapter<u8>* choice,
            const TensorAdapter<int64_t>* val_t,
            const TensorAdapter<int64_t>* val_f,
            TensorAdapter<int64_t>* ret) {
    auto send_ot = [](const TensorAdapter<int64_t>* diff,
                          const TensorBlock* round_ot_mask,
                          TensorAdapter<int64_t>* send_buffer,
                          TensorAdapter<int64_t>* ret) {
        //block round_ot_mask = recv_val<block>();

        // bad naming from ot extention
        auto q = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        ot()->ot_sender().get_ot_instance(q.get());

        //q ^= (round_ot_mask & ot()->base_ot_choice());
        auto base_ot_choice = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        ot()->base_ot_choice(base_ot_choice.get());
        auto tmp = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        round_ot_mask->bitwise_and(base_ot_choice.get(), tmp.get());
        q->bitwise_xor(tmp.get(), q.get());

        //auto s = common::hash_blocks({q, q ^ ot()->base_ot_choice()});
        auto s_first = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        auto s_second = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        q->bitwise_xor(base_ot_choice.get(), tmp.get());
        std::pair<TensorBlock*, TensorBlock*> x_pair({q.get(), tmp.get()});
        std::pair<TensorBlock*, TensorBlock*> s_pair({s_first.get(), s_second.get()});
        common::hash_blocks(x_pair, s_pair);

        //int64_t  s0 = *reinterpret_cast<int64_t *>(&s.first);
        TensorAdapter<int64_t>* s0 = ret;
        block_to_int64(s_first.get(), s0);

        //int64_t msg1 = diff ^ s0;
        auto msg1 = tensor_factory()->template create<int64_t>(s0->shape());
        diff->bitwise_xor(s0, msg1.get());

        //int64_t  s1 = *reinterpret_cast<int64_t *>(&s.second);
        auto s1 = tensor_factory()->template create<int64_t>(send_buffer->shape());
        block_to_int64(s_second.get(), s1.get());

        //s1 ^= msg1;
        s1->bitwise_xor(msg1.get(), s1.get());

        //send_to_buffer(s1);
        //send_buffer.emplace_back(s1);
        s1->copy(send_buffer);

        //return s0;
    };

    auto send_ot_mask = [](const TensorAdapter<u8>* choice, TensorBlock* send_buffer,
                           TensorAdapter<int64_t>* ret) {
        auto block_shape = get_block_shape(ret->shape());
        auto ot_ins0 = tensor_factory()->template create<int64_t>(block_shape);
        auto ot_ins1 = tensor_factory()->template create<int64_t>(block_shape);
        ot()->ot_receiver().get_ot_instance(ot_ins0.get(), ot_ins1.get());
        //block choice_ = choice ? common::OneBlock : common::ZeroBlock;
        auto choice_ = tensor_factory()->template create<int64_t>(block_shape);
        block* choice_ptr = reinterpret_cast<block*>(choice_->data());
        std::transform(choice->data(), choice->data() + choice->numel(),
                       choice_ptr, [](bool val) {
                           return val ? common::OneBlock : common::ZeroBlock;
                        });

        //const auto& t0 = ot_instance[0];
        //auto ot_mask = choice_ ^ ot_instance[0] ^ ot_instance[1];
        const auto& t0 = ot_ins0;
        auto ot_mask = tensor_factory()->template create<int64_t>(block_shape);
        choice_->bitwise_xor(ot_ins0.get(), ot_mask.get());
        ot_mask->bitwise_xor(ot_ins1.get(), ot_mask.get());

        //send_to_buffer(ot_mask);
        //send_buffer.emplace_back(ot_mask);
        ot_mask->copy(send_buffer);

        //auto t0_hash = common::hash_block(t0);
        //int64_t key = *reinterpret_cast<int64_t *>(&t0_hash);
        auto t0_hash = tensor_factory()->template create<int64_t>(block_shape);
        common::hash_block(t0.get(), t0_hash.get());
        block_to_int64(t0_hash.get(), ret);

        //return key;
    };

    auto recv_ot = [](const TensorAdapter<u8>* choice, TensorAdapter<int64_t>* key,
                      const TensorAdapter<int64_t>* round_ot_msg,
                      TensorAdapter<int64_t>* ret) {
        //int64_t ot_msg = recv_val<int64_t>();

        //return choice ? round_ot_msg ^ key : key;
        auto tmp = tensor_factory()->template create<int64_t>(key->shape());
        round_ot_msg->bitwise_xor(key, tmp.get());
        if_then_else_plain(choice, tmp.get(), key, ret);

    };

    //std::vector<int64_t> delta;
    //std::vector<int64_t> msg;
    //std::vector<int64_t> key;

    //std::vector<int64_t> send_buffer;
    //std::vector<block> send_buffer1;
    //std::vector<block> recv_buffer;
    //std::vector<int64_t> recv_buffer1;
    //recv_buffer.resize(size);
    //recv_buffer1.resize(size);
    auto shape = ret->shape();
    auto block_shape = get_block_shape(shape);
    auto delta = tensor_factory()->template create<int64_t>(shape);
    auto msg = tensor_factory()->template create<int64_t>(shape);
    auto key = tensor_factory()->template create<int64_t>(shape);
    auto send_buffer = tensor_factory()->template create<int64_t>(shape);
    auto send_buffer1 = tensor_factory()->template create<int64_t>(block_shape);
    auto recv_buffer = tensor_factory()->template create<int64_t>(block_shape);
    auto recv_buffer1 = tensor_factory()->template create<int64_t>(shape);

    if (party() == 0) {
        //net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(block));
        net()->recv(next_party(), *recv_buffer);
        //for (size_t i = 0; i < size; ++i) {
        //    const block& round_ot_mask = recv_buffer.at(i);
        //    delta.emplace_back(send_ot(val_t[i] ^ val_f[i], round_ot_mask, send_buffer));
        //}
        const auto& round_ot_mask = recv_buffer;
        auto tmp = tensor_factory()->template create<int64_t>(shape);
        val_t->bitwise_xor(val_f, tmp.get());
        send_ot(tmp.get(), round_ot_mask.get(), send_buffer.get(), delta.get());
        //flush_buffer();
        //net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(int64_t));
        net()->send(next_party(), *send_buffer);
        //for (size_t i = 0; i < size; ++i) {
        //    key.emplace_back(send_ot_mask(choice[i], send_buffer1));
        //}
        send_ot_mask(choice, send_buffer1.get(), key.get());
        //flush_buffer();
        //net()->send(next_party(), send_buffer1.data(), send_buffer1.size() * sizeof(block));
        //net()->recv(next_party(), recv_buffer1.data(), recv_buffer1.size() * sizeof(int64_t));
        net()->send(next_party(), *send_buffer1);
        net()->recv(next_party(), *recv_buffer1);
        //for (size_t i = 0; i < size; ++i) {
        //    const int64_t& round_ot_msg = recv_buffer1.at(i);
        //    msg.emplace_back(recv_ot(choice[i], key[i], round_ot_msg));
        //}
        const auto& round_ot_msg = recv_buffer1;
        recv_ot(choice, key.get(), round_ot_msg.get(), msg.get());
    } else {
        //for (size_t i = 0; i < size; ++i) {
        //    key.emplace_back(send_ot_mask(choice[i], send_buffer1));
        //}
        send_ot_mask(choice, send_buffer1.get(), key.get());
        //flush_buffer();
        //net()->send(next_party(), send_buffer1.data(), send_buffer1.size() * sizeof(block));
        //net()->recv(next_party(), recv_buffer1.data(), recv_buffer1.size() * sizeof(int64_t));
        net()->send(next_party(), *send_buffer1);
        net()->recv(next_party(), *recv_buffer1);
        //for (size_t i = 0; i < size; ++i) {
        //    const int64_t& round_ot_msg = recv_buffer1.at(i);
        //    msg.emplace_back(recv_ot(choice[i], key[i], round_ot_msg));
        //}
        const auto& round_ot_msg = recv_buffer1;
        recv_ot(choice, key.get(), round_ot_msg.get(), msg.get());
        //net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(block));
        net()->recv(next_party(), *recv_buffer);
        //for (size_t i = 0; i < size; ++i) {
        //    const block& round_ot_mask = recv_buffer.at(i);
        //    delta.emplace_back(send_ot(val_t[i] ^ val_f[i], round_ot_mask, send_buffer));
        //}
        const auto& round_ot_mask = recv_buffer;
        auto tmp = tensor_factory()->template create<int64_t>(shape);
        val_t->bitwise_xor(val_f, tmp.get());
        send_ot(tmp.get(), round_ot_mask.get(), send_buffer.get(), delta.get());
        //flush_buffer();
        ///net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(int64_t));
        net()->send(next_party(), *send_buffer);
    }

    //std::vector<int64_t> ret;
    //for (size_t i = 0; i < size; ++i) {
    //    ret.emplace_back(val_f[i] ^ delta[i] ^ msg[i] ^ choice[i] * (val_t[i] ^ val_f[i]));
    //}
    val_f->bitwise_xor(delta.get(), ret);
    ret->bitwise_xor(msg.get(), ret);
    //ret->bitwise_xor(choice, ret);
    std::transform(ret->data(), ret->data() + ret->numel(),
                    choice->data(), ret->data(), [](int64_t a, u8 b) {
                        return a ^ b;
                    });
    auto tmp = tensor_factory()->template create<int64_t>(shape);
    val_t->bitwise_xor(val_f, tmp.get());
    ret->mul(tmp.get(), ret);

    //return ret;
}

/*
int64_t bc_mux(bool choice, int64_t val_t, int64_t val_f) {
    uint8_t c = choice;
    return bc_mux_internal(&c, &val_t, &val_f, 1)[0];
}

std::vector<int64_t> bc_mux(const std::vector<uint8_t>& choice,
                                                  const std::vector<int64_t>& val_t,
                                                  const std::vector<int64_t>& val_f) {
    return bc_mux_internal(choice.data(), val_t.data(), val_f.data(), choice.size());
}
*/
}; // namespace smc
