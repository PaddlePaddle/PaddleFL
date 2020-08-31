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
#include "core/privc3/ot.h"

namespace aby3 {

template<typename T>
size_t BooleanTensor<T>::pre_party() const {
    return aby3_ctx()->pre_party();
}

template<typename T>
size_t BooleanTensor<T>::next_party() const {
    return aby3_ctx()->next_party();
}

template<typename T>
size_t BooleanTensor<T>::party() const {
    return aby3_ctx()->party();
}

template<typename T>
BooleanTensor<T>::BooleanTensor(TensorAdapter<T>* tensor[2]) {
    // TODO: check if tensor shape equal
    _share[0] = tensor[0];
    _share[1] = tensor[1];
}

template<typename T>
BooleanTensor<T>::BooleanTensor(TensorAdapter<T>* tensor0,
                                TensorAdapter<T>* tensor1) {
    // TODO: check if tensor shape equal
    _share[0] = tensor0;
    _share[1] = tensor1;
}

template<typename T>
BooleanTensor<T>::BooleanTensor() {
}

template<typename T>
TensorAdapter<T>* BooleanTensor<T>::share(size_t idx) {
    // TODO: check if idx < 2
    return _share[idx];
}

template<typename T>
const TensorAdapter<T>* BooleanTensor<T>::share(size_t idx) const {
    // TODO: check if idx < 2
    return _share[idx];
}

template<typename T>
void BooleanTensor<T>::reveal_to_one(size_t party_num, TensorAdapter<T>* ret) const {

    if (party_num == party()) {
        // TODO: check if tensor shape equal

        // incase of this and ret shares tensor ptr
        auto buffer = tensor_factory()->template create<T>(ret->shape());
        aby3_ctx()->network()->template recv(pre_party(), *buffer);

        share(0)->bitwise_xor(buffer.get(), ret);
        share(1)->bitwise_xor(ret, ret);

    } else if (party_num == next_party()) {

        aby3_ctx()->network()->template send(party_num, *share(0));

    }
}

template<typename T>
void BooleanTensor<T>::reveal(TensorAdapter<T>* ret) const {
    for (size_t idx = 0; idx < 3; ++idx) {
        reveal_to_one(idx, ret);
    }
}

template<typename T>
const std::vector<size_t> BooleanTensor<T>::shape() const {
    if (share(0)) {
        return share(0)->shape();
    }
    else {
        return std::vector<size_t>();
    }
}

template<typename T>
size_t BooleanTensor<T>::numel() const {
    if (share(0)) {
        return share(0)->numel();
    }
    else {
        0;
    }
}

template<typename T>
void BooleanTensor<T>::bitwise_xor(const BooleanTensor* rhs,
                                   BooleanTensor* ret) const {
    share(0)->bitwise_xor(rhs->share(0), ret->share(0));
    share(1)->bitwise_xor(rhs->share(1), ret->share(1));
}

template<typename T>
void BooleanTensor<T>::bitwise_xor(const TensorAdapter<T>* rhs,
                                   BooleanTensor* ret) const {
    share(0)->bitwise_xor(rhs, ret->share(0));
    share(1)->bitwise_xor(rhs, ret->share(1));
}

template<typename T>
void BooleanTensor<T>::bitwise_and(const BooleanTensor* rhs,
                                   BooleanTensor* ret) const {

    auto tmp_zero = tensor_factory()->template create<T>(ret->shape());
    auto tmp0 = tensor_factory()->template create<T>(ret->shape());
    auto tmp1 = tensor_factory()->template create<T>(ret->shape());
    auto tmp2 = tensor_factory()->template create<T>(ret->shape());

    aby3_ctx()->template gen_zero_sharing_boolean(*tmp_zero.get());

    share(0)->bitwise_and(rhs->share(0), tmp0.get());
    share(0)->bitwise_and(rhs->share(1), tmp1.get());
    share(1)->bitwise_and(rhs->share(0), tmp2.get());

    tmp0->bitwise_xor(tmp1.get(), tmp0.get());
    tmp0->bitwise_xor(tmp2.get(), tmp0.get());
    tmp0->bitwise_xor(tmp_zero.get(), ret->share(0));

    // 3-party msg send recv sequence
    //       p0      p1      p2
    // t0:  0->2            2<-0
    // t1:          1<-2    2->1
    // t2:  0<-1    1->2
    if (party() > 0) {
        aby3_ctx()->network()->template recv(next_party(), *(ret->share(1)));
        aby3_ctx()->network()->template send(pre_party(), *(ret->share(0)));
    } else {
        aby3_ctx()->network()->template send(pre_party(), *(ret->share(0)));
        aby3_ctx()->network()->template recv(next_party(), *(ret->share(1)));
    }
}

template<typename T>
void BooleanTensor<T>::bitwise_and(const TensorAdapter<T>* rhs,
                                   BooleanTensor* ret) const {
    share(0)->bitwise_and(rhs, ret->share(0));
    share(1)->bitwise_and(rhs, ret->share(1));
}

template<typename T>
template<template<typename U> class CTensor>
void BooleanTensor<T>::bitwise_or(const CTensor<T>* rhs,
                                  BooleanTensor* ret) const {

    std::vector<std::shared_ptr<TensorAdapter<T>>> tmp;

    for (int i = 0; i < 2; ++i) {
        tmp.emplace_back(
            tensor_factory()->template create<T>(shape()));
    }

    BooleanTensor buffer(tmp[0].get(), tmp[1].get());
    // ret = x & y
    bitwise_and(rhs, &buffer);
    // ret = x & y ^ x
    bitwise_xor(&buffer, &buffer);
    // ret = x & y ^ x ^ y
    buffer.bitwise_xor(rhs, ret);
}

template<typename T>
void BooleanTensor<T>::bitwise_not(BooleanTensor* ret) const {
    if (party() == 0) {
        share(0)->bitwise_not(ret->share(0));
        share(1)->copy(ret->share(1));
    } else if (party() == 1) {
        share(0)->copy(ret->share(0));
        share(1)->copy(ret->share(1));
    } else {
        share(0)->copy(ret->share(0));
        share(1)->bitwise_not(ret->share(1));
    }
}

template<typename T>
void BooleanTensor<T>::lshift(size_t rhs, BooleanTensor* ret) const {
    share(0)->lshift(rhs, ret->share(0));
    share(1)->lshift(rhs, ret->share(1));
}

template<typename T>
void BooleanTensor<T>::rshift(size_t rhs, BooleanTensor* ret) const {
    share(0)->rshift(rhs, ret->share(0));
    share(1)->rshift(rhs, ret->share(1));
}

template<typename T>
void BooleanTensor<T>::logical_rshift(size_t rhs, BooleanTensor* ret) const {
    share(0)->logical_rshift(rhs, ret->share(0));
    share(1)->logical_rshift(rhs, ret->share(1));
}

template<typename T>
void BooleanTensor<T>::ppa(const BooleanTensor* rhs,
                           BooleanTensor* ret,
                           size_t n_bits) const {
    // kogge stone adder from tfe
    // https://github.com/tf-encrypted
    // TODO: check T is int64_t other native type not support yet
    const size_t k = std::ceil(std::log2(n_bits));
    std::vector<T> keep_masks(k);
    for (size_t i = 0; i < k; ++i) {
        keep_masks[i] = (T(1) << (T) std::exp2(i)) - 1;
    }

    std::shared_ptr<TensorAdapter<T>> tmp[11];
    for (auto& ti: tmp) {
        ti = tensor_factory()->template create<T>(ret->shape());
    }
    BooleanTensor<T> g(tmp[0].get(), tmp[1].get());
    BooleanTensor<T> p(tmp[2].get(), tmp[3].get());
    BooleanTensor<T> g1(tmp[4].get(), tmp[5].get());
    BooleanTensor<T> p1(tmp[6].get(), tmp[7].get());
    BooleanTensor<T> c(tmp[8].get(), tmp[9].get());
    auto k_mask = tmp[10].get();

    bitwise_and(rhs, &g);
    bitwise_xor(rhs, &p);

    for (size_t i = 0; i < k; ++i) {

        std::transform(k_mask->data(), k_mask->data() + k_mask->numel(),
                       k_mask->data(),
                       [&keep_masks, i](T) -> T { return keep_masks[i]; });

        g.lshift(std::exp2(i), &g1);
        p.lshift(std::exp2(i), &p1);


        p1.bitwise_xor(k_mask, &p1);
        g1.bitwise_and(&p, &c);

        g.bitwise_xor(&c, &g);
        p.bitwise_and(&p1, &p);
    }
    g.lshift(1, &c);
    bitwise_xor(rhs, &p);

    c.bitwise_xor(&p, ret);
}

template<typename T, size_t N>
void a2b(AbstractContext* aby3_ctx,
         TensorAdapterFactory* tensor_factory,
         const FixedPointTensor<T, N>* a,
         BooleanTensor<T>* b,
         size_t n_bits) {

    std::shared_ptr<TensorAdapter<T>> tmp[4];
    for (auto& ti: tmp) {
        ti = tensor_factory->template create<T>(a->shape());
        // set 0
        std::transform(ti->data(), ti->data() + ti->numel(), ti->data(),
                       [](T) -> T { return 0; });
    }

    std::shared_ptr<BooleanTensor<T>> lhs =
            std::make_shared<BooleanTensor<T>>(tmp[0].get(), tmp[1].get());
    std::shared_ptr<BooleanTensor<T>> rhs =
            std::make_shared<BooleanTensor<T>>(tmp[2].get(), tmp[3].get());

    if (aby3_ctx->party() == 0) {
        a->share(0)->add(a->share(1), lhs->share(0));

        // reshare x0 + x1
        aby3_ctx->template gen_zero_sharing_boolean(*lhs->share(1));
        lhs->share(0)->bitwise_xor(lhs->share(1), lhs->share(0));

        aby3_ctx->network()->template send(2, *(lhs->share(0)));
        aby3_ctx->network()->template recv(1, *(lhs->share(1)));

    } else if (aby3_ctx->party() == 1) {

        aby3_ctx->template gen_zero_sharing_boolean(*lhs->share(0));
        aby3_ctx->network()->template send(0, *(lhs->share(0)));
        aby3_ctx->network()->template recv(2, *(lhs->share(1)));

        a->share(1)->copy(rhs->share(1));

    } else { // party == 2

        aby3_ctx->template gen_zero_sharing_boolean(*lhs->share(0));

        aby3_ctx->network()->template recv(0, *(lhs->share(1)));
        aby3_ctx->network()->template send(1, *(lhs->share(0)));

        a->share(0)->copy(rhs->share(0));
    }

    lhs->ppa(rhs.get(), b, n_bits);
}

template<typename T>
template<size_t N>
BooleanTensor<T>& BooleanTensor<T>::operator=(const FixedPointTensor<T, N>* other) {
    a2b(aby3_ctx().get(), tensor_factory().get(), other, this, sizeof(T) * 8);
    return *this;
}

template <typename T>
void tensor_rshift_transform(const TensorAdapter<T>* lhs,
                             size_t rhs, TensorAdapter<T>* ret) {
    const T* begin = lhs->data();
    std::transform(begin, begin + lhs->numel(), ret->data(),
                   [rhs](T in) { return (in >> rhs) & 1; });
};

template<typename T>
template<size_t N>
void BooleanTensor<T>::bit_extract(size_t i, const FixedPointTensor<T, N>* in) {
    a2b(aby3_ctx().get(), tensor_factory().get(), in, this, i + 1);

    tensor_rshift_transform(share(0), i, share(0));
    tensor_rshift_transform(share(1), i, share(1));
}

template<typename T>
void BooleanTensor<T>::bit_extract(size_t i, BooleanTensor* ret) const {
    tensor_rshift_transform(share(0), i, ret->share(0));
    tensor_rshift_transform(share(1), i, ret->share(1));
}

template<typename T>
template<size_t N>
void BooleanTensor<T>::b2a(FixedPointTensor<T, N>* ret) const {
    std::shared_ptr<TensorAdapter<T>> tmp[2];
    for (auto& ti: tmp) {
        ti = tensor_factory()->template create<T>(shape());
        // set 0
        std::transform(ti->data(), ti->data() + ti->numel(), ti->data(),
                       [](T) -> T { return 0; });
    }
    BooleanTensor<T> bt(tmp[0].get(), tmp[1].get());

    if (party() == 1) {
        aby3_ctx()->template gen_random(*ret->mutable_share(0), 0);
        aby3_ctx()->template gen_random(*ret->mutable_share(1), 1);
        ret->share(0)->add(ret->share(1), tmp[0].get());
        tmp[0]->negative(tmp[0].get());
        aby3_ctx()->network()->template send(0, *(tmp[0].get()));
    } else if (party() == 0) {
        aby3_ctx()->network()->template recv(1, *(tmp[1].get()));
        // dummy gen random, for prng sync
        aby3_ctx()->template gen_random(*ret->mutable_share(1), 1);
    } else { // party == 2
        aby3_ctx()->template gen_random(*ret->mutable_share(0), 0);
    }

    bt.ppa(this, &bt, sizeof(T) * 8);

    TensorAdapter<T>* dest = nullptr;
    if (party() == 0) {
        dest =  ret->mutable_share(0);
    }

    bt.reveal_to_one(0, dest);

    if (party() == 0) {
        aby3_ctx()->network()->template recv(1, *(ret->mutable_share(1)));
        aby3_ctx()->network()->template send(2, *(ret->mutable_share(0)));
    } else if (party() == 1) {
        aby3_ctx()->network()->template send(0, *(ret->mutable_share(0)));
    } else { // party == 2
        aby3_ctx()->network()->template recv(0, *(ret->mutable_share(1)));
    }
}

template<typename T>
template<size_t N>
void BooleanTensor<T>::mul(const TensorAdapter<T>* rhs,
                           FixedPointTensor<T, N>* ret,
                           size_t rhs_party) const {
    // ot sender
    size_t idx0 = rhs_party;

    size_t idx1 = (rhs_party + 1) % 3;

    size_t idx2 = (rhs_party + 2) % 3;

    auto tmp0 = tensor_factory()->template create<T>(ret->shape());
    auto tmp1 = tensor_factory()->template create<T>(ret->shape());

    TensorAdapter<T>* tmp[2] = {tmp0.get(), tmp1.get()};

    TensorAdapter<T>* null_arg[2] = {nullptr, nullptr};

    if (party() == idx0) {
        // use ret as buffer
        TensorAdapter<T>* m[2] = {ret->mutable_share(0), ret->mutable_share(1)};

        aby3_ctx()->template gen_zero_sharing_arithmetic(*tmp[0]);

        // m0 = a * (b0 ^ b1) + s0
        // m1 = a * (1 ^ b0 ^ b1) + s0
        share(0)->bitwise_xor(share(1), m[0]);
        std::transform(m[0]->data(), m[0]->data() + m[0]->numel(), m[0]->data(),
                       [](T in) { return 1 & in; });
        std::transform(m[0]->data(), m[0]->data() + m[0]->numel(), m[1]->data(),
                       [](T in) { return 1 ^ in; });

        m[0]->mul(rhs, m[0]);
        m[1]->mul(rhs, m[1]);

        m[0]->add(tmp[0], m[0]);
        m[1]->add(tmp[0], m[1]);

        ObliviousTransfer::ot(idx0, idx1, idx2, null_arg[0],
                    const_cast<const aby3::TensorAdapter<T>**>(m),
                    tmp, null_arg[0]);

        // ret0 = s2
        // ret1 = s1
        aby3_ctx()->network()->template recv(idx2, *(ret->mutable_share(0)));
        aby3_ctx()->network()->template recv(idx1, *(ret->mutable_share(1)));

    } else if (party() == idx1) {
        // ret0 = s1
        aby3_ctx()->template gen_zero_sharing_arithmetic(*(ret->mutable_share(0)));
        // ret1 = a * b + s0
        ObliviousTransfer::ot(idx0, idx1, idx2, share(1),
                    const_cast<const aby3::TensorAdapter<T>**>(null_arg),
                    tmp, ret->mutable_share(1));
        aby3_ctx()->network()->template send(idx0, *(ret->share(0)));
        aby3_ctx()->network()->template send(idx2, *(ret->share(1)));
    } else if (party() == idx2) {
        // ret0 = a * b + s0
        aby3_ctx()->template gen_zero_sharing_arithmetic(*(ret->mutable_share(1)));
        // ret1 = s2
        ObliviousTransfer::ot(idx0, idx1, idx2, share(0),
                    const_cast<const aby3::TensorAdapter<T>**>(null_arg),
                    tmp, null_arg[0]);

        aby3_ctx()->network()->template send(idx0, *(ret->share(1)));

        aby3_ctx()->network()->template recv(idx1, *(ret->mutable_share(0)));
    }
}

template<typename T>
template<size_t N>
void BooleanTensor<T>::mul(const FixedPointTensor<T, N>* rhs,
                           FixedPointTensor<T, N>* ret) const {
    std::vector<std::shared_ptr<TensorAdapter<T>>> tmp;

    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(
            tensor_factory()->template create<T>(ret->shape()));
    }

    FixedPointTensor<T, N> tmp0(tmp[0].get(), tmp[1].get());
    FixedPointTensor<T, N> tmp1(tmp[2].get(), tmp[3].get());

    if (party() == 0) {
        mul(nullptr, &tmp0, 1);
        mul(rhs->share(0), &tmp1, 0);
    } else if (party() == 1) {
        rhs->share(0)->add(rhs->share(1), tmp[2].get());
        mul(tmp[2].get(), &tmp0, 1);
        mul(nullptr, &tmp1, 0);
    } else { // party() == 2
        mul(nullptr, &tmp0, 1);
        mul(nullptr, &tmp1, 0);
    }
    tmp0.add(&tmp1, ret);
}
template<typename T>
void BooleanTensor<T>::onehot_from_cmp() {
    // cmp is done slice by slice
    // suppose that shape = [k, m, n, ...]
    // shape of all slices and tmp tensors = [1, m, n]
    auto shape_ = shape();
    size_t len = shape_[0];
    shape_[0] = 1;
    std::vector<std::shared_ptr<TensorAdapter<T>>> tmp;

    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(
            tensor_factory()->template create<T>(shape_));
    }

    tmp.emplace_back(tensor_factory()->template create<T>());
    tmp.emplace_back(tensor_factory()->template create<T>());

    BooleanTensor found(tmp[0].get(), tmp[1].get());

    assign_to_tensor(tmp[0].get(), T(0));
    assign_to_tensor(tmp[1].get(), T(0));

    BooleanTensor not_found(tmp[2].get(), tmp[3].get());

    // res[i] = !found & input[i]
    // found = found 1 res[i]
    // to find last 1, we search backward
    for (size_t i = len; i > 0; --i) {
        share(0)->slice(i - 1, i, tmp[4].get());
        share(1)->slice(i - 1, i, tmp[5].get());
        BooleanTensor cmp_i(tmp[4].get(), tmp[5].get());
        found.bitwise_not(&not_found);
        not_found.bitwise_and(&cmp_i, &cmp_i);
        cmp_i.bitwise_or(&found, &found);
    }
}
} // namespace aby3
