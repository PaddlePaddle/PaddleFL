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

#include "gc_bit.h"
#include "paddle/fluid/platform/enforce.h"

namespace privc {

inline void to_bits(const TensorAdapter<int64_t>* input,
                    TensorAdapter<u8>* input_bits) {
    // change integer tensor to bits tensor
    PADDLE_ENFORCE_EQ(input_bits->shape()[0],
                      sizeof(int64_t) * 8,  // 1 byte = 8 bits
                      "shape error, first shape of return should be %s",
                      sizeof(int64_t) * 8);
    PADDLE_ENFORCE_EQ(input_bits->numel(),
                      input->numel() * sizeof(int64_t) * 8,  // 1 byte = 8 bits
                      "input shape error.");
    for (int i = 0; i < sizeof(int64_t) * 8; ++i) {
        auto input_slice = tensor_factory()->template create<u8>(input->shape());
        input_bits->slice(i, i + 1, input_slice.get());

        std::transform(input->data(), input->data() + input->numel(),
                        input_slice->data(),
                        [&i](int64_t a) {
                            u8 val = (a >> i) & (u8) 1;
                            return val;
                        });
    }
}

void to_ac_num(const TensorAdapter<int64_t>* val,
               TensorAdapter<int64_t>* ret);

void bc_mux(const TensorAdapter<u8>* choice,
            const TensorAdapter<int64_t>* val_t,
            const TensorAdapter<int64_t>* val_f,
            TensorAdapter<int64_t>* ret);

template<size_t N>
class GCFixedPointTensor {
protected:
    int _length;
    std::shared_ptr<TensorBlock> _bits_tensor;

public:

    ~GCFixedPointTensor() {}

    GCFixedPointTensor(const TensorAdapter<int64_t>* input, size_t party);

    GCFixedPointTensor() : _length(0), _bits_tensor(nullptr) {}
    GCFixedPointTensor(const std::vector<size_t>& shape) : _length(shape[0]) {
        _bits_tensor = tensor_factory()->template create<int64_t>(shape);
        std::for_each(_bits_tensor->data(), _bits_tensor->data() + _bits_tensor->numel(),
                      [](int64_t& a) { a = 0; });
    }

    GCFixedPointTensor(double in, std::vector<size_t> shape) {
        auto gc_shape = get_gc_shape(shape);
        _length = gc_shape[0];
        _bits_tensor = tensor_factory()->template create<int64_t>(gc_shape);
        std::for_each(_bits_tensor->data(), _bits_tensor->data() + _bits_tensor->numel(),
                      [](int64_t& a) { a = 0; });
        int64_t in_ = (int64_t) (in * std::pow(2, N));
        for (int i = 0; i < _length; i += 1) {
            if (party() == 0 && in_ >> i & 1) {
                auto share_i = (*this)[i];
                auto garbled_delta = tensor_factory()->template create<int64_t>(share_i->shape());
                ot()->garbled_delta(garbled_delta.get());
                garbled_delta->copy(share_i->mutable_share());
            }
        }
    }

    std::vector<size_t> shape() const {
        return _bits_tensor->shape();
    }

    // Comparable
    void geq(const GCFixedPointTensor* rhs, BitTensor* ret) const;
    void equal(const GCFixedPointTensor* rhs, BitTensor* ret) const;

    inline int size() const { return _length; }

    TensorBlock* mutable_share() { return _bits_tensor.get(); }
    const TensorBlock* share() const { return _bits_tensor.get(); }

    void bitwise_add(const GCFixedPointTensor* rhs, GCFixedPointTensor* ret) const;
    void bitwise_sub(const GCFixedPointTensor* rhs, GCFixedPointTensor* ret) const;
    void bitwise_neg(GCFixedPointTensor* ret) const;
    void bitwise_mul(const GCFixedPointTensor* rhs, GCFixedPointTensor* ret) const;
    void bitwise_div(const GCFixedPointTensor* rhs, GCFixedPointTensor* ret) const;
    void bitwise_xor(const GCFixedPointTensor* rhs, GCFixedPointTensor* ret) const;

    void abs(GCFixedPointTensor* ret) const;

    // get index bits, i.e., bit_tensor[idx]
    std::shared_ptr<BitTensor> operator[](int index);
    const std::shared_ptr<BitTensor> operator[](int index) const;

    void reconstruct(TensorAdapter<int64_t>* ret) const {
        PADDLE_ENFORCE_EQ(share()->numel() / shape()[0] / shape()[1],
                      ret->numel(), "input numel no match.");

        lsb(ret);
        auto remote = tensor_factory()->template create<int64_t>(ret->shape());

        if (party() == 0) {
            net()->send(next_party(), *ret);

            net()->recv(next_party(), *remote);
            ret->bitwise_xor(remote.get(), ret);
        } else {
            net()->recv(next_party(), *remote);
            net()->send(next_party(), *ret);

            ret->bitwise_xor(remote.get(), ret);
        }
    }

    void is_zero(BitTensor* ret) const;

    void lsb(TensorAdapter<int64_t>* ret) const {
        // get least significate bit
        PADDLE_ENFORCE_EQ(share()->numel() / shape()[0] / shape()[1],
                      ret->numel(), "input numel no match.");

        std::for_each(ret->data(), ret->data() + ret->numel(),
                      [](int64_t& a) { a = 0;});
        for (int idx = 0; idx < size(); idx += 1) {
            auto tmp = tensor_factory()->template create<int64_t>(ret->shape());
            block_lsb((*this)[idx]->share(), tmp.get());

            tmp->lshift(idx, tmp.get());

            ret->bitwise_or(tmp.get(), ret);
        }
    }

    static void if_then_else(BitTensor* cond, const GCFixedPointTensor* t_int,
                             const GCFixedPointTensor* f_int, GCFixedPointTensor* ret);

    static void if_then_else_bc(BitTensor* cond, const GCFixedPointTensor* t_int,
                                const GCFixedPointTensor* f_int, TensorAdapter<int64_t>* ret);

    // with return ciphertext of one-hot
    static void argmax_one_hot(const GCFixedPointTensor* op, GCFixedPointTensor* ret);

    void relu(GCFixedPointTensor* ret) const;

    void relu_bc(TensorAdapter<int64_t>* ret) const;

    void logistic(GCFixedPointTensor* ret) const;

};

template<size_t N>
void if_then_else(GCFixedPointTensor<N>* dest, const GCFixedPointTensor<N> *tsrc,
                  const GCFixedPointTensor<N> *fsrc, int size,
                  BitTensor* cond, int pos_dest = 0,
                  int pos_tsrc = 0, int pos_fsrc = 0);

template<size_t N>
void mul_full(GCFixedPointTensor<N> *dest, const GCFixedPointTensor<N> *op1,
              const GCFixedPointTensor<N> *op2, int size);

template<size_t N>
void cond_neg(BitTensor* cond, GCFixedPointTensor<N> *dest,
              const GCFixedPointTensor<N> *src);

template<size_t N>
void div_full(GCFixedPointTensor<N> *vquot, GCFixedPointTensor<N> *vrem,
              const GCFixedPointTensor<N> *op1, const GCFixedPointTensor<N> *op2);

} // namespace privc

#include "gc_fixedpoint_tensor_imp.h"
