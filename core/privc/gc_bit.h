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
#include "core/common/crypto.h"
#include "core/privc/triplet_generator.h"
#include "core/privc/common_utils.h"
#include "core/privc/ot.h"
#include "paddle/fluid/platform/enforce.h"

namespace privc {

void garbled_and(const TensorBlock* a, const TensorBlock* b, TensorBlock* ret);

void garbled_share(const TensorAdapter<u8>* val, TensorBlock* ret);

template<typename T>
inline void if_then_else_plain(const TensorAdapter<T>* val,
                         const TensorBlock* then_val,
                         const TensorBlock* else_val,
                         TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(_g_block_size_expand * val->numel(),
                      then_val->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(else_val->numel(), then_val->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(ret->numel(), then_val->numel(),
                      "input numel no match.");

    const block* then_val_ptr = reinterpret_cast<const block*>(then_val->data());
    const block* else_val_ptr = reinterpret_cast<const block*>(else_val->data());
    block* ret_ptr = reinterpret_cast<block*>(ret->data());
    for (int i = 0; i < val->numel(); ++i) {
        *(ret_ptr + i) = *(val->data() + i) ?
                         *(then_val_ptr + i) : *(else_val_ptr + i);
    }
}

template<typename T>
inline void if_then_else_plain(bool is_block_val,
                         const TensorAdapter<T>* val,
                         const TensorAdapter<int64_t>* then_val,
                         const TensorAdapter<int64_t>* else_val,
                         TensorAdapter<int64_t>* ret) {
    if (is_block_val) {
        if_then_else_plain(val, then_val, else_val, ret);
    } else {
        for (int i = 0; i < val->numel(); ++i) {
            *(ret->data() + i) = *(val->data() + i) ? 
                                 *(then_val->data() + i) :
                                 *(else_val->data() + i);
        }
    }
}

class BitTensor {
private:
    // using double memory of TensorAdapter<int64_t> to present
    // TensorAdapter<block>
    std::shared_ptr<TensorBlock> _share;

public:
    BitTensor() = delete;
    BitTensor(std::shared_ptr<TensorBlock> share) {
        _share = share;
    }

    BitTensor(std::vector<size_t> shape) {
        _share = tensor_factory()->template create<int64_t>(shape);
        std::for_each(_share->data(), _share->data() + _share->numel(),
                      [](int64_t& a) { a = 0;});
    }

    BitTensor(TensorAdapter<u8>* val, size_t party_in) {
        auto block_shape = val->shape();
        block_shape.insert(block_shape.begin(), 2);
        _share = tensor_factory()->template create<int64_t>(block_shape);
        if (party_in == 0) {
            if (party() == 0) {
                privc_ctx()->template gen_random_private(*_share);

                auto to_send = tensor_factory()->template create<int64_t>(block_shape);
                _share->copy(to_send.get());
                auto mask_to_send = tensor_factory()->template create<int64_t>(block_shape);
                auto garbled_delta = tensor_factory()->template create<int64_t>(block_shape);
                ot()->garbled_delta(garbled_delta.get());
                to_send->bitwise_xor(garbled_delta.get(), mask_to_send.get());
                if_then_else_plain(val, mask_to_send.get(), to_send.get(), to_send.get());

                net()->send(next_party(), *to_send);

            } else {
                // maybe need to know recv shape
                net()->recv(next_party(), *_share);
            }
        } else {
            garbled_share(val, _share.get());
        }
    }

    ~BitTensor() {}

    std::vector<size_t> shape() const {
        return _share->shape();
    }
    void set_false() {
        std::for_each(_share->data(), _share->data() + _share->numel(),
                      [](int64_t& a) { a = 0; });
    }

    void bitwise_xor(const BitTensor* rhs, BitTensor* ret) const {
        PADDLE_ENFORCE_EQ(rhs->share()->numel(), share()->numel(),
                          "input numel no match.");
        PADDLE_ENFORCE_EQ(ret->share()->numel(), share()->numel(),
                          "input numel no match.");
        share()->bitwise_xor(rhs->share(), ret->mutable_share());
    }

    const TensorBlock* share() const {
        return _share.get();
    }

    TensorBlock* mutable_share() {
        return _share.get();
    }

    void bitwise_and(const BitTensor* rhs, BitTensor* ret) const {
        PADDLE_ENFORCE_EQ(rhs->share()->numel(), share()->numel(),
                          "input numel no match.");
        PADDLE_ENFORCE_EQ(ret->share()->numel(), share()->numel(),
                          "input numel no match.");

        garbled_and(share(), rhs->share(), ret->mutable_share());
    }

    void bitwise_or(const BitTensor* rhs, BitTensor* ret) const {
        PADDLE_ENFORCE_EQ(rhs->share()->numel(), share()->numel(),
                          "input numel no match.");
        PADDLE_ENFORCE_EQ(ret->share()->numel(), share()->numel(),
                          "input numel no match.");

        BitTensor op_t(shape());
        bitwise_and(rhs, &op_t);
        bitwise_xor(&op_t, &op_t);
        rhs->bitwise_xor(&op_t, ret);
    }

    void bitwise_not(BitTensor* ret) const {
        PADDLE_ENFORCE_EQ(ret->share()->numel(), share()->numel(),
                          "input numel no match.");

        share()->copy(ret->mutable_share());
        auto garbled_delta = tensor_factory()->template create<int64_t>(shape());
        ot()->garbled_delta(garbled_delta.get());
        if (party() == 0) {
            ret->share()->bitwise_xor(garbled_delta.get(), ret->mutable_share());
        }
    }

    void lsb(TensorAdapter<u8>* ret) {
        PADDLE_ENFORCE_EQ(ret->numel() * _g_block_size_expand,
                          share()->numel(), "input numel no match.");
        block_lsb(share(), ret);
    }

    void reconstruct(TensorAdapter<u8>* ret) const {
        PADDLE_ENFORCE_EQ(ret->numel() * _g_block_size_expand,
                          share()->numel(), "input numel no match.");
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

} // namespace privc

