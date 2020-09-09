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

namespace privc {

void add_full(Bit *dest, Bit *carry_out, const Bit *op1, const Bit *op2,
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
}

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
}

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
}

void if_then_else(Bit *dest, const Bit *tsrc, const Bit *fsrc, int size,
                  Bit cond) {
    int i = 0;
    while (size-- > 0) {
        auto x = tsrc[i] ^ fsrc[i];
        auto a = cond & x;
        dest[i] = a ^ fsrc[i];
        ++i;
    }
}

void cond_neg(Bit cond, Bit *dest, const Bit *src, int size) {
    int i = 0;
    Bit c = cond;
    for (i = 0; i < size - 1; ++i) {
        dest[i] = src[i] ^ cond;
        Bit t = dest[i] ^ c;
        c = c & dest[i];
        dest[i] = t;
    }
    dest[i] = cond ^ c ^ src[i];
}

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
}

Integer::Integer(int64_t input, size_t party_in)
    : _length(sizeof(input) * 8), _bits(_length) {
    if (party_in == 0) {
        if (party() == 0) {
            std::vector<block> send_buffer;
            for (u64 idx = 0; idx < sizeof(input) * 8; idx += 1) {

                block to_send = privc_ctx()->gen_random_private<block>();
                _bits[idx]._share = to_send;

                to_send ^=
                    (input >> idx) & 1 ? ot()->garbled_delta() : psi::ZeroBlock;
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

std::vector<Integer> Integer::vector(const std::vector<int64_t>& input,
                                     size_t party_in) {
    const size_t length = sizeof(int64_t) * 8;
    std::vector<Integer> ret;
    if (party_in == 0) {
        if (party() == 0) {
            std::vector<block> send_buffer;
            for (size_t i = 0; i < input.size(); ++i) {
                Integer int_i;
                int_i._length = length;
                int_i._bits.resize(length);
                for (u64 idx = 0; idx < length; idx += 1) {

                    block to_send = privc_ctx()->gen_random_private<block>();
                    int_i._bits[idx]._share = to_send;

                    to_send ^=
                        (input[i] >> idx) & 1 ? ot()->garbled_delta() : psi::ZeroBlock;
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
                Integer int_i;
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
            Integer int_i;
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

Bit &Integer::operator[](int index) {
    return _bits[std::min(index, size() - 1)];
}

const Bit &Integer::operator[](int index) const {
    return _bits[std::min(index, size() - 1)];
}

// Comparisons
Bit Integer::geq(const Integer &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    Integer tmp = (*this) - rhs;

    std::vector<Bit> dest(size());
    Bit borrow_out;

    sub_full(dest.data(), &borrow_out, cbits(), rhs.cbits(), nullptr, size());
    return ~(borrow_out ^ _bits[size() - 1] ^ rhs._bits[size() - 1]);
}

Bit Integer::equal(const Integer &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    Bit res;
    res = ~res;
    for (int i = 0; i < size(); ++i) {
        res = res & ~(_bits[i] ^ rhs._bits[i]);
    }
    return res;
}

Bit Integer::is_zero() const {
    Bit res;
    for (int i = 0; i < size(); ++i) {
        res = res | _bits[i];
    }
    return ~res;
}

Integer Integer::abs() const {
    Integer res(*this);
    for (int i = 0; i < size(); ++i) {
        res[i] = _bits[size() - 1];
    }
    return ((*this) + res) ^ res;
}

Integer Integer::operator^(const Integer &rhs) const {
    Integer res(*this);
    for (int i = 0; i < size(); ++i) {
        res._bits[i] = res._bits[i] ^ rhs._bits[i];
    }
    return res;
}

// Arithmethics
Integer Integer::operator+(const Integer &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    Integer res(*this);
    add_full(res.bits(), nullptr, cbits(), rhs.cbits(), nullptr, size());
    return res;
}

Integer Integer::operator-(const Integer &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    Integer res(*this);
    sub_full(res.bits(), nullptr, cbits(), rhs.cbits(), nullptr, size());
    return res;
}

Integer Integer::operator*(const Integer &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    Integer res(*this);
    mul_full(res.bits(), cbits(), rhs.cbits(), size());
    return res;
}

Integer Integer::operator/(const Integer &rhs) const {
    if (size() != rhs.size()) {
        throw std::logic_error("op len not match");
    }
    Integer res(*this);
    Integer i1 = abs();
    Integer i2 = rhs.abs();
    Bit sign = _bits[size() - 1] ^ rhs[size() - 1];
    div_full(res.bits(), nullptr, i1.cbits(), i2.cbits(), size());
    Bit q_sign = res[size() - 1];
    std::vector<Bit> nan(size(), q_sign);
    nan[size() - 1] = ~q_sign;

    privc::if_then_else(res.bits(), nan.data(), res.cbits(), size(), q_sign);

    cond_neg(sign, res.bits(), res.cbits(), size());
    res[0] = res[0] ^ (sign & q_sign);
    return res;
}

Integer Integer::operator-() const {
    std::vector<Bit> zero(size());
    Integer res(*this);
    sub_full(res.bits(), nullptr, zero.data(), cbits(), nullptr, size());
    return res;
}

Integer Integer::if_then_else(Bit cond, const Integer &t_int,
                              const Integer &f_int) {
    Integer res(t_int);
    privc::if_then_else(res.bits(), t_int.cbits(), f_int.cbits(), res.size(), cond);

    return res;
}

int64_t Integer::if_then_else_bc(Bit cond, const Integer &t_int,
                                const Integer &f_int) {
    return bc_mux(block_lsb(cond._share), t_int.lsb(), f_int.lsb());
}

int64_t Integer::argmax(const std::vector<Integer>& op, size_t party_in) {
    size_t size = op.size();
    if (size <= 0) {
        throw std::logic_error("op len error");
    }
    if (size == 1) {
        return 0;
    }
    std::vector<Bit> cmp_bit;
    cmp_bit.resize(size - 1);

    Integer max = op[0];
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

std::vector<int64_t> Integer::argmax_one_hot(
                const std::vector<Integer>& op) {
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

    Integer max = op[0];
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

    // convert Bit to Integer
    ret.resize(res_b.size());
    for( int i = 0; i < ret.size(); ++i) {
      ret[i] = (int64_t)res_b[i].lsb();
    }
    return ret;
}

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

                auto s = psi::hash_blocks({q, q ^ ot()->base_ot_choice()});
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

                auto t0_hash = psi::hash_block(t0_buffer[i * word_width + idx]);

                int64_t key = *reinterpret_cast<int64_t *>(&t0_hash);
                ret[i] += (input[i] >> idx) & 1 ? round_ot_msg ^ key : key;
                ot_msg_idx++;
            }
        }

    }

    return ret;
}

int64_t to_ac_num(int64_t val) {
    return to_ac_num_internal(&val, 1)[0];
}

std::vector<int64_t> to_ac_num(const std::vector<int64_t>& input) {
    return to_ac_num_internal(input.data(), input.size());
}

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

        auto s = psi::hash_blocks({q, q ^ ot()->base_ot_choice()});
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
        block choice_ = choice ? psi::OneBlock : psi::ZeroBlock;

        const auto& t0 = ot_instance[0];
        auto ot_mask = choice_ ^ ot_instance[0] ^ ot_instance[1];

        //send_to_buffer(ot_mask);
        send_buffer.emplace_back(ot_mask);

        auto t0_hash = psi::hash_block(t0);
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

int64_t bc_mux(bool choice, int64_t val_t, int64_t val_f) {
    uint8_t c = choice;
    return bc_mux_internal(&c, &val_t, &val_f, 1)[0];
}

std::vector<int64_t> bc_mux(const std::vector<uint8_t>& choice,
                                                  const std::vector<int64_t>& val_t,
                                                  const std::vector<int64_t>& val_f) {
    return bc_mux_internal(choice.data(), val_t.data(), val_f.data(), choice.size());
}

}; // namespace smc
