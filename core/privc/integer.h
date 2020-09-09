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

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "bit.h"

namespace privc {

int64_t to_ac_num(int64_t val);

std::vector<int64_t> to_ac_num(const std::vector<int64_t>& input);

int64_t bc_mux(bool choice, int64_t val_t, int64_t val_f);

std::vector<int64_t> bc_mux(const std::vector<uint8_t>& choice,
                                const std::vector<int64_t>& val_t,
                                const std::vector<int64_t>& val_f);

class Integer {
protected:
    int _length;
    std::vector<Bit> _bits;

public:
    Integer(Integer &&in) : _length(in._length) {
        std::swap(_bits, in._bits);
    }

    Integer(const Integer &in)
        : _length(in._length), _bits(in._bits) {}

    Integer &operator=(Integer &&rhs) {
        _length = rhs._length;
        std::swap(_bits, rhs._bits);
        return *this;
    }

    Integer &operator=(const Integer &rhs) {
        _length = rhs._length;
        _bits = rhs._bits;
        return *this;
    }

    ~Integer() {}

    Integer(int64_t input, size_t party);

    static std::vector<Integer> vector(const std::vector<int64_t>& input,
                                       size_t party);

    Integer() : _length(0), _bits() {}

    // Comparable
    Bit geq(const Integer &rhs) const;
    Bit equal(const Integer &rhs) const;

    inline int size() const { return _length; }

    Bit* bits() { return _bits.data(); }
    const Bit* cbits() const { return _bits.data(); }

    std::vector<Bit>& share() { return _bits; }
    const std::vector<Bit>& share() const { return _bits; }

    Integer operator+(const Integer &rhs) const;
    Integer operator-(const Integer &rhs) const;
    Integer operator-() const;
    Integer operator*(const Integer &rhs) const;
    Integer operator/(const Integer &rhs) const;
    Integer operator^(const Integer &rhs) const;

    Integer abs() const;

    Bit& operator[](int index);
    const Bit& operator[](int index) const;

    int64_t reconstruct() const {
        int64_t ret = lsb();

        if (party() == 0) {
            net()->send(next_party(), ret);
            ret ^= net()->recv<int64_t>(next_party());
        } else {
            auto remote = net()->recv<int64_t>(next_party());
            net()->send(next_party(), ret);
            ret ^= remote;
        }

        return ret;
    }

    bool reconstruct(u64 idx) const {

        if (idx >= (unsigned)size()) {
            throw std::logic_error("vector range exceed");
        }

        auto bit = _bits[idx].reconstruct();
        return bit ? 1 : 0;
    }

    Bit is_zero() const;

    int64_t lsb() const {
        int64_t ret = 0;
        for (int idx = 0; idx < size(); idx += 1) {
            ret |= (int64_t)block_lsb(_bits[idx]._share) << idx;
        }
        return ret;
    }

    static Integer if_then_else(Bit cond, const Integer &t_int,
                                const Integer &f_int);

    static int64_t if_then_else_bc(Bit cond, const Integer &t_int,
                                     const Integer &f_int);

    // input one dimension, return plaintext
    static int64_t argmax(const std::vector<Integer>& op,
                          size_t party = std::numeric_limits<size_t>::max());
    // with return ciphertext of one-hot
    static std::vector<int64_t> argmax_one_hot(
                          const std::vector<Integer>& op);

};

void if_then_else(Bit *dest, const Bit *tsrc, const Bit *fsrc, int size,
                  Bit cond);

void cond_neg(Bit cond, Bit *dest, const Bit *src, int size);

void mul_full(Bit *dest, const Bit *op1, const Bit *op2, int size);

void div_full(Bit *vquot, Bit *vrem, const Bit *op1, const Bit *op2, int size);

typedef Integer Int64gc;

} // namespace privc

