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

#include <memory>
#include <algorithm>

#include "paddle/fluid/platform/enforce.h"
#include "../privc3/prng.h"

namespace privc {

template<typename T, size_t N>
FixedPointTensor<T, N>::FixedPointTensor(TensorAdapter<T>* share_tensor) {
    _share = share_tensor;
}

template<typename T, size_t N>
TensorAdapter<T>* FixedPointTensor<T, N>::mutable_share() {
    return _share;
}

template<typename T, size_t N>
const TensorAdapter<T>* FixedPointTensor<T, N>::share() const {
    return _share;
}

// reveal fixedpointtensor to one party
template<typename T, size_t N>
void FixedPointTensor<T, N>::reveal_to_one(size_t party,
                                           TensorAdapter<T>* ret) const {

    if (party == this->party()) {
        auto buffer = tensor_factory()->template create<T>(ret->shape());
        privc_ctx()->network()->template recv(next_party(), *buffer);

        share()->add(buffer.get(), ret);
        ret->scaling_factor() = N;
    } else {
        privc_ctx()->network()->template send(party, *share());
    }
}

// reveal fixedpointtensor to all parties
template<typename T, size_t N>
void FixedPointTensor<T, N>::reveal(TensorAdapter<T>* ret) const {
    for (size_t i = 0; i < 2; ++i) {
        reveal_to_one(i, ret);
    }
}

template<typename T, size_t N>
const std::vector<size_t> FixedPointTensor<T, N>::shape() const {
    return _share->shape();
}

//convert TensorAdapter to shares
template<typename T, size_t N>
void FixedPointTensor<T, N>::share(const TensorAdapter<T>* input,
                                    TensorAdapter<T>* output_shares[2],
                                    block seed) {

    if (psi::equals(seed, psi::g_zero_block)) {
        seed = psi::block_from_dev_urandom();
    }
    //set seed of prng[2]
    privc_ctx()->set_random_seed(seed, 2);

    privc_ctx()->template gen_random_private(*output_shares[0]);

    input->sub(output_shares[0], output_shares[1]);
    for (int i = 0; i < 2; ++i) {
        output_shares[i]->scaling_factor() = input->scaling_factor();
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::add(const FixedPointTensor<T, N>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    _share->add(rhs->_share, ret->_share);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::add(const TensorAdapter<T>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    if (party() == 0) {
        _share->add(rhs, ret->_share);
    } else {
        _share->copy(ret->_share);
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sub(const FixedPointTensor<T, N>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    _share->sub(rhs->_share, ret->_share);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sub(const TensorAdapter<T>* rhs,
                                FixedPointTensor<T, N>* ret) const {
    if (party() == 0) {
        _share->sub(rhs, ret->_share);
    } else {
        _share->copy(ret->_share);
    }
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::negative(FixedPointTensor<T, N>* ret) const {
    _share->negative(ret->_share);
}

template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::mul_impl(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret,
                                 const Type2Type<int64_t>) const {
    auto triplet_shape = shape();
    triplet_shape.insert(triplet_shape.begin(), 3);
    auto triplet = tensor_factory()->template create<T>(triplet_shape);
    tripletor()->get_triplet(triplet.get());

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (int i = 0; i < 8; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>(ret->shape()));
    }
    FixedPointTensor<T, N> a(temp[0].get());
    FixedPointTensor<T, N> b(temp[1].get());
    FixedPointTensor<T, N> c(temp[2].get());
    auto parse_triplet = [&triplet](int idx, FixedPointTensor<T, N>& ret) {
      triplet->slice(idx, idx + 1, ret.mutable_share());
      auto shape = ret.shape();
      shape.erase(shape.begin());
      ret.mutable_share()->reshape(shape);
    };

    parse_triplet(0, a);
    parse_triplet(1, b);
    parse_triplet(2, c);

    FixedPointTensor<T, N> e(temp[3].get());
    FixedPointTensor<T, N> f(temp[4].get());
    this->sub(&a, &e);
    rhs->sub(&b, &f);

    auto& reveal_e = temp[5];
    auto& reveal_f = temp[6];

    e.reveal(reveal_e.get());
    f.reveal(reveal_f.get());

    FixedPointTensor<T, N> ft_temp(temp[7].get());
    fixed64_tensor_mult<N>(reveal_f.get(), a.share(), ret->mutable_share());
    fixed64_tensor_mult<N>(reveal_e.get(), b.share(), ft_temp.mutable_share());

    ret->add(&ft_temp, ret);
    ret->add(&c, ret);

    if(party() == 1) {
        auto& ef = temp[7];
        ef->scaling_factor() = N;
        fixed64_tensor_mult<N>(reveal_e.get(), reveal_f.get(), ef.get());
        ret->share()->add(ef.get(), ret->mutable_share());
    }
}

} // namespace privc
