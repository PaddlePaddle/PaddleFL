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
#include <memory>

#include "bit.h"

namespace privc {

inline void to_bits(const TensorAdapter<int64_t>* input, TensorAdapter<u8>* input_bits) {

    for (int i = 0; i < sizeof(int64_t) * 8; ++i) {
        //auto tmp = tensor_factory()->template create<int64_t>(input->shape());
        //input->rshift(i, tmp.get());
        // TODO: make sure deconstructing input_slice has not effect on input_bits
        auto input_slice = tensor_factory()->template create<u8>(input->shape());
        //TensorAdapter<u8>* input_slice;
        input_bits->slice(i, i + 1, input_slice.get());
        //tmp->bitwise_and(tensor_one.get(), input_slice.get());
        std::transform(input->data(), input->data() + input->numel(),
                        input_slice->data(),
                        [&i](int64_t a) {
                            u8 val = (a >> i) & (u8) 1;
                            return val;
                        });
    }
}
/*
int64_t to_ac_num(int64_t val);

std::vector<int64_t> to_ac_num(const std::vector<int64_t>& input);

int64_t bc_mux(bool choice, int64_t val_t, int64_t val_f);

std::vector<int64_t> bc_mux(const std::vector<uint8_t>& choice,
                                const std::vector<int64_t>& val_t,
                                const std::vector<int64_t>& val_f);
                                */
void to_ac_num(const TensorAdapter<int64_t>* val, TensorAdapter<int64_t>* ret);

void bc_mux(const TensorAdapter<u8>* choice,
            const TensorAdapter<int64_t>* val_t,
            const TensorAdapter<int64_t>* val_f,
            TensorAdapter<int64_t>* ret);

class IntegerTensor {
protected:
    int _length;
    std::shared_ptr<TensorBlock> _bits_tensor;

public:
    IntegerTensor(IntegerTensor &&in) : _length(in._length) {
        std::swap(_bits_tensor, in._bits_tensor);
    }

    IntegerTensor(const IntegerTensor &in)
        : _length(in._length) {
        _bits_tensor = tensor_factory()->template create<int64_t>(in._bits_tensor->shape());
        in._bits_tensor->copy(_bits_tensor.get());
    }

    IntegerTensor &operator=(IntegerTensor &&rhs) {
        _length = rhs._length;
        std::swap(_bits_tensor, rhs._bits_tensor);
        return *this;
    }

    IntegerTensor &operator=(const IntegerTensor &rhs) {
        _length = rhs._length;
        _bits_tensor = tensor_factory()->template create<int64_t>(rhs._bits_tensor->shape());
        rhs._bits_tensor->copy(_bits_tensor.get());
        return *this;
    }

    ~IntegerTensor() {}

    //Integer(int64_t input, size_t party);

    //static std::vector<Integer> vector(const std::vector<int64_t>& input,
    //                                   size_t party);
    IntegerTensor(const TensorAdapter<int64_t>* input, size_t party);

    IntegerTensor() : _length(0), _bits_tensor(nullptr) {}
    IntegerTensor(const std::vector<size_t>& shape) : _length(shape[0]) {
        _bits_tensor = tensor_factory()->template create<int64_t>(shape);
        std::for_each(_bits_tensor->data(), _bits_tensor->data() + _bits_tensor->numel(),
                      [](int64_t& a) { a = 0; });
    }

    std::vector<size_t> shape() const {
        return _bits_tensor->shape();
    }

    // Comparable
    void geq(const IntegerTensor* rhs, BitTensor* ret) const;
    void equal(const IntegerTensor* rhs, BitTensor* ret) const;

    inline int size() const { return _length; }

    //Bit* bits() { return _bits.data(); }
    //const Bit* cbits() const { return _bits.data(); }

    TensorBlock* mutable_share() { return _bits_tensor.get(); }
    const TensorBlock* share() const { return _bits_tensor.get(); }

    /*Integer operator+(const Integer &rhs) const;
    Integer operator-(const Integer &rhs) const;
    Integer operator-() const;
    Integer operator*(const Integer &rhs) const;
    Integer operator/(const Integer &rhs) const;
    Integer operator^(const Integer &rhs) const;*/
    void bitwise_add(const IntegerTensor* rhs, IntegerTensor* ret) const;
    void bitwise_sub(const IntegerTensor* rhs, IntegerTensor* ret) const;
    void bitwise_neg(IntegerTensor* ret) const;
    void bitwise_mul(const IntegerTensor* rhs, IntegerTensor* ret) const;
    void bitwise_xor(const IntegerTensor* rhs, IntegerTensor* ret) const;

    void abs(IntegerTensor* ret) const;

    std::shared_ptr<BitTensor> operator[](int index);
    const std::shared_ptr<BitTensor> operator[](int index) const;
/*
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
    }*/

    void reconstruct(TensorAdapter<int64_t>* ret) const {
        //int64_t ret = lsb();
        lsb(ret);
        auto remote = tensor_factory()->template create<int64_t>(ret->shape());

        if (party() == 0) {
            net()->send(next_party(), *ret);
            //ret ^= net()->recv<int64_t>(next_party());
            net()->recv(next_party(), *remote);
            ret->bitwise_xor(remote.get(), ret);
        } else {
            //auto remote = net()->recv<int64_t>(next_party());
            net()->recv(next_party(), *remote);
            net()->send(next_party(), *ret);
            //ret ^= remote;
            ret->bitwise_xor(remote.get(), ret);
        }

        //return ret;
    }

    /*bool reconstruct(u64 idx) const {

        if (idx >= (unsigned)size()) {
            throw std::logic_error("vector range exceed");
        }

        auto bit = _bits[idx].reconstruct();
        return bit ? 1 : 0;
    }*/

    void is_zero(BitTensor* ret) const;

/*    int64_t lsb() const {
        int64_t ret = 0;
        for (int idx = 0; idx < size(); idx += 1) {
            ret |= (int64_t)block_lsb(_bits[idx]._share) << idx;
        }
        return ret;
    }*/

    void lsb(TensorAdapter<int64_t>* ret) const {
        //int64_t ret = 0;
        std::for_each(ret->data(), ret->data() + ret->numel(),
                      [](int64_t& a) { a = 0;});
        for (int idx = 0; idx < size(); idx += 1) {
            //ret |= (int64_t)block_lsb(_bits[idx]._share) << idx;
            auto tmp = tensor_factory()->template create<int64_t>(ret->shape());
            block_lsb((*this)[idx]->share(), tmp.get());
            //std::cout <<"p0: "<<tmp.get();
            tmp->lshift(idx, tmp.get());
            //std::cout <<"p1"<<tmp.get();
            ret->bitwise_or(tmp.get(), ret);
        }
        //return ret;
    }

//    static Integer if_then_else(Bit cond, const Integer &t_int,
//                                const Integer &f_int);

    static void if_then_else(BitTensor* cond, const IntegerTensor* t_int,
                             const IntegerTensor* f_int, IntegerTensor* ret);

//    static int64_t if_then_else_bc(Bit cond, const Integer &t_int,
//                                     const Integer &f_int);

    static void if_then_else_bc(BitTensor* cond, const IntegerTensor* t_int,
                                const IntegerTensor* f_int, TensorAdapter<int64_t>* ret);
    // input one dimension, return plaintext
    //static int64_t argmax(const std::vector<Integer>& op,
    //                      size_t party = std::numeric_limits<size_t>::max());
    // with return ciphertext of one-hot
    //static std::vector<int64_t> argmax_one_hot(
    //                      const std::vector<Integer>& op);

};

//void if_then_else(Bit *dest, const Bit *tsrc, const Bit *fsrc, int size,
//                  Bit cond);

void if_then_else(IntegerTensor* dest, const IntegerTensor *tsrc, const IntegerTensor *fsrc, int size,
                  BitTensor* cond, IntegerTensor* ret);

//void cond_neg(Bit cond, Bit *dest, const Bit *src, int size);

void mul_full(IntegerTensor *dest, const IntegerTensor *op1, const IntegerTensor *op2, int size);

//void div_full(Bit *vquot, Bit *vrem, const Bit *op1, const Bit *op2, int size);

//typedef Integer Int64gc;

} // namespace privc

