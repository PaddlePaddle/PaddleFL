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

#include <vector>
#include <limits>

#include "core/privc/privc_context.h"
#include "core/privc/crypto.h"
#include "core/privc/triplet_generator.h"
#include "core/privc/common_utils.h"
#include "core/privc/ot.h"

namespace privc {
/*
block garbled_and(block a, block b);

block garbled_share(bool val);

std::vector<block> garbled_share(int64_t val);

std::vector<block> garbled_share(const std::vector<int64_t>& val);
*/
void garbled_and(const TensorBlock* a, const TensorBlock* b, TensorBlock* ret);

void garbled_share(const TensorAdapter<u8>* val, TensorBlock* ret);

template<typename T>
inline void if_than_else_plain(const TensorAdapter<T>* val,
                         const TensorBlock* than_val,
                         const TensorBlock* else_val,
                         TensorBlock* ret) {
    const block* than_val_ptr = reinterpret_cast<const block*>(than_val->data());
    const block* else_val_ptr = reinterpret_cast<const block*>(else_val->data());
    block* ret_ptr = reinterpret_cast<block*>(ret->data());
    for (int i = 0; i < val->numel(); ++i) {
        *(ret_ptr + i) = *(val->data() + i) ? *(than_val_ptr + i) : *(else_val_ptr + i);
    }
}

class BitTensor {
public:
    // using double memory of TensorAdapter<int64_t> to present
    // TensorAdapter<block>
    std::shared_ptr<TensorBlock> _share;
    //block _share;

public:
    //Bit() : _share(ZeroBlock) {}
    BitTensor() = delete;
    BitTensor(std::shared_ptr<TensorBlock> share) {
        _share = share;
        std::for_each(_share->data(), _share->data() + _share->numel(),
                      [](int64_t& a) { a = 0;});
    }

    BitTensor(std::vector<size_t> shape) {
        _share = tensor_factory()->template create<int64_t>(shape);
        std::for_each(_share->data(), _share->data() + _share->numel(),
                      [](int64_t& a) { a = 0;});
    }

    BitTensor(BitTensor &&in) {
        std::swap(_share, in._share);
    }

    BitTensor(const BitTensor &in) {
        _share = tensor_factory()->template create<int64_t>(in.shape());
        in._share->copy(_share.get());
    }

    BitTensor &operator=(BitTensor &&rhs) {
        std::swap(_share, rhs._share);
        return *this;
    }

    BitTensor &operator=(const BitTensor &rhs) {
        _share = tensor_factory()->template create<int64_t>(rhs.shape());
        rhs._share->copy(_share.get());
        return *this;
    }

    /*Bit(bool val, size_t party_in) {
        if (party_in == 0) {
            if (party() == 0) {
                _share = privc_ctx()->gen_random_private<block>();
                block to_send = _share;
                if (val) {
                    to_send ^= ot()->garbled_delta();
                }
                net()->send(next_party(), to_send);

            } else {
                _share = net()->recv<block>(next_party());
            }
        } else {
            _share = garbled_share(val);
        }
    }*/

    BitTensor(TensorAdapter<u8>* val, size_t party_in) {
        auto block_shape = val->shape();
        block_shape.insert(block_shape.begin(), 2);
        if (party_in == 0) {
            if (party() == 0) {
                _share = tensor_factory()->template create<int64_t>(block_shape);
                privc_ctx()->template gen_random_private(*_share);
                //block to_send = _share;
                //if (val) {
                //    to_send ^= ot()->garbled_delta();
                //}
                // to_send = val * to_send + (1 - val) * mask_to_send
                auto to_send = tensor_factory()->template create<int64_t>(block_shape);
                _share->copy(to_send.get());
                auto mask_to_send = tensor_factory()->template create<int64_t>(block_shape);
                auto garbled_delta = tensor_factory()->template create<int64_t>(block_shape);
                to_send->bitwise_xor(garbled_delta.get(), mask_to_send.get());
                if_than_else_plain(val, mask_to_send.get(), to_send.get(), to_send.get());

                net()->send(next_party(), *to_send);

            } else {
                // need to know recv shape
                _share = tensor_factory()->template create<int64_t>(block_shape);
                net()->recv(next_party(), *_share);
            }
        } else {
            garbled_share(val, _share.get());
        }
    }

    ~BitTensor() {}

    /*Bit operator^(const Bit &rhs) const {
        Bit ret;
        ret._share = _share ^ rhs._share;
        return ret;
    }*/

    std::vector<size_t> shape() const {
        return _share->shape();
    }

    void bitwise_xor(const BitTensor* rhs, BitTensor* ret) const {
        share()->bitwise_xor(rhs->share(), ret->mutable_share());
    }

//    block& share() {
//      return _share;
//    }

    const TensorBlock* share() const {
        return _share.get();
    }

    //const block& share() const {
    //  return _share;
    //}

    TensorBlock* mutable_share() {
        return _share.get();
    }

    /*Bit operator&(const Bit &rhs) const {
        Bit ret;
        ret._share = garbled_and(_share, rhs._share);
        return ret;
    }*/
    void bitwise_and(const BitTensor* rhs, BitTensor* ret) const {
        garbled_and(share(), rhs->share(), ret->mutable_share());
    }

    //Bit operator|(const Bit &rhs) const { return *this ^ rhs ^ (*this & rhs); }
    void bitwise_or(const BitTensor* rhs, BitTensor* ret) const {
        BitTensor op_t(shape());
        bitwise_and(rhs, &op_t);
        bitwise_xor(&op_t, &op_t);
        rhs->bitwise_xor(&op_t, ret);
    }

    /*Bit operator~() const {
        Bit ret;

        ret._share = _share;

        if (party() == 0) {
            ret._share ^= ot()->garbled_delta();
        }

        return ret;
    }*/
    void bitwise_not(BitTensor* ret) const {
        share()->copy(ret->mutable_share());
        auto garbled_delta = tensor_factory()->template create<int64_t>(shape());
        ot()->garbled_delta(garbled_delta.get());
        if (party() == 0) {
            ret->share()->bitwise_xor(garbled_delta.get(), ret->mutable_share());
        }
    }

    /*Bit operator&&(const Bit &rhs) const {
        return *this & rhs;
    }

    Bit operator||(const Bit &rhs) const {
        return *this | rhs;
    }

    Bit operator!() const {
        return ~*this;
    }
    */

    //u8 lsb() const {
    //  u8 ret = block_lsb(_share);
    //  return ret & (u8)1;
    //}
    void lsb(TensorAdapter<u8>* ret) {
        block_lsb(share(), ret);
    }

    /*bool reconstruct() const {
        u8 remote;
        u8 local = block_lsb(_share);

        if (party() == 0) {
            remote = net()->recv<u8>(next_party());
            net()->send(next_party(), local);
        } else {
            net()->send(next_party(), local);
            remote = net()->recv<u8>(next_party());
        }

        return remote ^ local;
    }*/

    void reconstruct(TensorAdapter<u8>* ret) const {
        std::vector<size_t> shape = this->shape();
        shape.erase(shape.begin());
        auto remote = tensor_factory()->template create<u8>(shape);
        auto local = tensor_factory()->template create<u8>(shape);
        block_lsb(share(), local.get());

        if (party() == 0) {
            net()->recv(next_party(), *remote);
            net()->send(next_party(), *local);
        } else {
            net()->send(next_party(), *local);
            net()->recv(next_party(), *remote);
        }
        remote->bitwise_xor(local.get(), ret);
    }

};

//std::vector<bool> reconstruct(std::vector<Bit> bits,
//              size_t party_in);

//using Bool = Bit;

} // namespace privc

