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
#include "../privc3/paddle_tensor.h"

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

template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::mul_impl(const TensorAdapter<T>* rhs,
                                 FixedPointTensor<T, N>* ret,
                                 const Type2Type<int64_t>) const {
    fixed64_tensor_mult<N>(share(), rhs, ret->mutable_share());
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::div(const TensorAdapter<T>* rhs,
                                 FixedPointTensor<T, N>* ret) const {

    auto temp = tensor_factory()->template create<T>(this->shape());

    double scale = std::pow(2, N);
    auto inverse = [scale](T d) -> T {
                    return 1.0 * scale / d * scale; };
    std::transform(rhs->data(), rhs->data() + rhs->numel(),
                                temp->data(), inverse);

    this->mul(temp.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::sum(FixedPointTensor* ret) const {
    PADDLE_ENFORCE_EQ(ret->numel(), 1, "output size should be 1.");
    T sum = (T) 0;
    for (int i = 0; i < numel(); ++i) {
        sum += *(share()->data() + i);
    }
    *(ret->mutable_share()->data()) = sum;
}

template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::mat_mul_impl(const FixedPointTensor<T, N>* rhs,
                                 FixedPointTensor<T, N>* ret,
                                 const Type2Type<int64_t>) const {
    // A * B, assume A.shape = [a, b], B.shape = [b, c]
    size_t row = ret->shape()[0];
    size_t col = ret->shape()[1];
    PADDLE_ENFORCE_EQ(row, shape()[0], "invalid result shape for mat mul");
    PADDLE_ENFORCE_EQ(col, rhs->shape()[1], "invalid result shape for mat mul");
    PADDLE_ENFORCE_EQ(shape()[1], rhs->shape()[0], "invalid input shape for mat mul");

    //transpose rhs
    auto shape_trans = rhs->shape();
    std::swap(shape_trans[0], shape_trans[1]);
    auto trans_rhs = tensor_factory()->template create<T>(shape_trans);
    const aby3::PaddleTensor<T>* p_rhs = dynamic_cast<const aby3::PaddleTensor<T>*>(rhs->share());
    const_cast<aby3::PaddleTensor<T>*>(p_rhs)->template Transpose<2>(std::vector<int>({1, 0}), trans_rhs.get());

    //get penta triplet, shape = [5, a, c, b]
    std::vector<size_t> penta_triplet_shape{5, shape()[0], shape_trans[0], shape_trans[1]};
    auto penta_triplet = tensor_factory()->template create<T>(penta_triplet_shape);
    tripletor()->get_penta_triplet(penta_triplet.get());

    // get triplet[idx0][idx1][idx2], shape = [b]
    auto access_triplet = [&penta_triplet, &penta_triplet_shape](size_t idx0,
                                                                 size_t idx1,
                                                                 size_t idx2,
                                                                 TensorAdapter<T>* ret) {
      size_t numel = penta_triplet->numel();
      auto& shape = penta_triplet_shape;
      int64_t* tripl_ptr = penta_triplet->data();
      size_t cal_idx_begin = idx0 * numel / shape[0]
                             + idx1 * numel / (shape[0] * shape[1])
                             + idx2 * numel / (shape[0] * shape[1] * shape[2]);
      std::copy(tripl_ptr + cal_idx_begin,
                tripl_ptr + cal_idx_begin + shape[3],
                ret->data());
    };

    auto slice_and_reshape = [](const TensorAdapter<T>* input, int idx, TensorAdapter<T>* ret) {
      input->slice(idx, idx + 1, ret);
      auto shape = ret->shape();
      shape.erase(shape.begin());
      ret->reshape(shape);
    };

    std::vector<int64_t> buffer_e;
    std::vector<int64_t> buffer_f;
    buffer_e.resize(col * row * shape()[1]);
    buffer_f.resize(col * row * shape()[1]);
    int64_t* buffer_e_ptr = buffer_e.data();
    int64_t* buffer_f_ptr = buffer_f.data();

    // cal share <e>, <f>
    for (int i = 0; i < row; ++i) {
      auto lhs_v = tensor_factory()->template create<T>({shape()[1]});
      slice_and_reshape(share(), i, lhs_v.get());

      for (int j = 0; j < col; ++j) {
        std::vector<size_t> shape_v{ shape()[1] };
        std::vector<std::shared_ptr<TensorAdapter<T>>> temp_triplet_i_j;
        for (int k = 0; k < 5; ++k) {
            temp_triplet_i_j.emplace_back(
                tensor_factory()->template create<T>(shape_v));
        }

        auto& a_i_j = temp_triplet_i_j[0];
        auto& alpha_i_j = temp_triplet_i_j[1];
        auto& b_i_j = temp_triplet_i_j[2];
        auto& c_i_j = temp_triplet_i_j[3];
        auto& alpha_c_i_j = temp_triplet_i_j[4];

        access_triplet(0, i, j / 2, a_i_j.get());
        access_triplet(1, i, j / 2, alpha_i_j.get());
        access_triplet(2, i, j / 2, b_i_j.get());
        access_triplet(3, i, j / 2, c_i_j.get());
        access_triplet(4, i, j / 2, alpha_c_i_j.get());

        auto e_v = tensor_factory()->template create<T>(shape_v);
        auto f_v = tensor_factory()->template create<T>(shape_v);

        auto rhs_v = tensor_factory()->template create<T>(shape_v);
        slice_and_reshape(trans_rhs.get(), j, rhs_v.get());
        if (j % 2 == 0) {
          lhs_v->sub(a_i_j.get(), e_v.get());
        } else {
          lhs_v->sub(alpha_i_j.get(), e_v.get());
        }
        rhs_v->sub(b_i_j.get(), f_v.get());

        std::copy(e_v->data(), e_v->data() + shape_v[0], buffer_e_ptr);
        std::copy(f_v->data(), f_v->data() + shape_v[0], buffer_f_ptr);
        buffer_e_ptr += shape_v[0];
        buffer_f_ptr += shape_v[0];
      }
    }

    // reveal all e and f
    std::vector<int64_t> remote_buffer_e;
    std::vector<int64_t> remote_buffer_f;
    remote_buffer_e.resize(col * row * shape()[1]);
    remote_buffer_f.resize(col * row * shape()[1]);
    if (party() == 0) {
      net()->send(next_party(), buffer_e.data(), buffer_e.size() * sizeof(int64_t));
      net()->send(next_party(), buffer_f.data(), buffer_f.size() * sizeof(int64_t));
      net()->recv(next_party(), remote_buffer_e.data(), remote_buffer_e.size() * sizeof(int64_t));
      net()->recv(next_party(), remote_buffer_f.data(), remote_buffer_f.size() * sizeof(int64_t));
    } else {
      net()->recv(next_party(), remote_buffer_e.data(), remote_buffer_e.size() * sizeof(int64_t));
      net()->recv(next_party(), remote_buffer_f.data(), remote_buffer_f.size() * sizeof(int64_t));
      net()->send(next_party(), buffer_e.data(), buffer_e.size() * sizeof(int64_t));
      net()->send(next_party(), buffer_f.data(), buffer_f.size() * sizeof(int64_t));
    }

    std::vector<int64_t> e;
    std::vector<int64_t> f;
    e.resize(col * row * shape()[1]);
    f.resize(col * row * shape()[1]);
    std::transform(buffer_e.begin(), buffer_e.end(),
                   remote_buffer_e.begin(), e.begin(),
                   std::plus<int64_t>());
    std::transform(buffer_f.begin(), buffer_f.end(),
                   remote_buffer_f.begin(), f.begin(),
                   std::plus<int64_t>());

    int64_t* e_ptr = e.data();
    int64_t* f_ptr = f.data();
    auto result = tensor_factory()->template create<T>(ret->shape());
    int64_t* res_ptr = result->data();

    // cal z = f<a> + e<b> + <c> or z = ef + f<a> + e<b> + <c>
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        std::vector<size_t> shape_v{ shape()[1] };
        std::vector<std::shared_ptr<TensorAdapter<T>>> temp_triplet_i_j;
        for (int k = 0; k < 5; ++k) {
            temp_triplet_i_j.emplace_back(
                tensor_factory()->template create<T>(shape_v));
        }

        auto& a_i_j = temp_triplet_i_j[0];
        auto& alpha_i_j = temp_triplet_i_j[1];
        auto& b_i_j = temp_triplet_i_j[2];
        auto& c_i_j = temp_triplet_i_j[3];
        auto& alpha_c_i_j = temp_triplet_i_j[4];

        access_triplet(0, i, j / 2, a_i_j.get());
        access_triplet(1, i, j / 2, alpha_i_j.get());
        access_triplet(2, i, j / 2, b_i_j.get());
        access_triplet(3, i, j / 2, c_i_j.get());
        access_triplet(4, i, j / 2, alpha_c_i_j.get());

        auto e_v = tensor_factory()->template create<T>(shape_v);
        auto f_v = tensor_factory()->template create<T>(shape_v);

        std::copy(e_ptr, e_ptr + shape_v[0], e_v->data());
        std::copy(f_ptr, f_ptr + shape_v[0], f_v->data());
        e_ptr += shape_v[0];
        f_ptr += shape_v[0];

        auto z_v = tensor_factory()->template create<T>(shape_v);
        fixed64_tensor_mult<N>(e_v.get(), b_i_j.get(), z_v.get());
        if (party() == 0) {
          auto ef = tensor_factory()->template create<T>(shape_v);
          fixed64_tensor_mult<N>(e_v.get(), f_v.get(), ef.get());
          z_v->add(ef.get(), z_v.get());
        }
        auto fa = tensor_factory()->template create<T>(shape_v);
        if (j % 2 == 0) {
          fixed64_tensor_mult<N>(f_v.get(), a_i_j.get(), fa.get());
          z_v->add(c_i_j.get(), z_v.get());
        } else {
          fixed64_tensor_mult<N>(f_v.get(), alpha_i_j.get(), fa.get());
          z_v->add(alpha_c_i_j.get(), z_v.get());
        }
        z_v->add(fa.get(), z_v.get());

        auto sum_v = [&z_v] () -> int64_t {
          int64_t sum = 0;
          for (int i = 0; i < z_v->numel(); ++i) {
            sum += *(z_v->data() + i);
          }
          return sum;
        };
        *res_ptr = sum_v();
        ++res_ptr;
      }
    }

    result->copy(ret->mutable_share());
}

} // namespace privc
