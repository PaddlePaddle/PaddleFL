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

#include "core/privc/triplet_generator.h"
#include "core/common/crypto.h"
#include "core/privc/privc_context.h"

namespace privc {

template<typename T, size_t N>
void TripletGenerator<T, N>::get_triplet(TensorAdapter<T>* ret) {
  size_t num_trip = ret->numel() / 3;
  if (_triplet_buffer.size() < num_trip) {
      fill_triplet_buffer();
  }

  for (int i = 0; i < num_trip; ++i) {
    auto triplet = _triplet_buffer.front();
    auto ret_ptr = ret->data() + i;
    *ret_ptr = triplet[0];
    *(ret_ptr + num_trip) = triplet[1];
    *(ret_ptr + 2 * num_trip) = triplet[2];
    _triplet_buffer.pop();
  }
}

template<typename T, size_t N>
void TripletGenerator<T, N>::get_penta_triplet(TensorAdapter<T>* ret) {
  size_t num_trip = ret->numel() / 5;
  if (_triplet_buffer.size() < num_trip) {
      fill_penta_triplet_buffer();
  }

  for (int i = 0; i < num_trip; ++i) {
    auto triplet = _penta_triplet_buffer.front();
    auto ret_ptr = ret->data() + i;
    *ret_ptr = triplet[0];
    *(ret_ptr + num_trip) = triplet[1];
    *(ret_ptr + 2 * num_trip) = triplet[2];
    *(ret_ptr + 3 * num_trip) = triplet[3];
    *(ret_ptr + 4 * num_trip) = triplet[4];
    _triplet_buffer.pop();
  }
}

template<typename T, size_t N>
template<typename T_>
void TripletGenerator<T, N>::fill_triplet_buffer_impl(const Type2Type<int64_t>) {
  std::vector<uint64_t> a(_s_triplet_step);
  std::vector<uint64_t> b(_s_triplet_step);

  std::for_each(a.data(), a.data() + a.size(),
                [this](uint64_t& val) {
                  val = this-> template gen_random_private<uint64_t>(); });
  std::for_each(b.data(), b.data() + b.size(),
                [this](uint64_t& val) {
                  val = this-> template gen_random_private<uint64_t>(); });

  std::vector<uint64_t> ab0;
  std::vector<uint64_t> ab1;

  std::function<std::vector<uint64_t>(const std::vector<uint64_t>&)> gen_p
  = [this](const std::vector<uint64_t>& v) {
      return gen_product(v);
  };

  ab0 = gen_p(this->party() == 0 ? a : b);
  ab1 = gen_p(this->party() == 0 ? b : a);

  for (uint64_t i = 0; i < a.size(); i += 1) {
    std::array<int64_t, 3> item = {
          static_cast<int64_t>(a[i]),
          static_cast<int64_t>(b[i]),
          static_cast<int64_t>(signedfixed128_mult<N>(a[i], b[i]) + ab0[i] + ab1[i])};
    _triplet_buffer.push(std::move(item));
  }
}

template<typename T, size_t N>
template<typename T_>
void TripletGenerator<T, N>::fill_penta_triplet_buffer_impl(const Type2Type<int64_t>) {
  std::vector<uint64_t> a(_s_triplet_step);
  std::vector<uint64_t> b(_s_triplet_step);
  std::vector<uint64_t> alpha(_s_triplet_step);

  std::for_each(a.data(), a.data() + a.size(),
                [this](uint64_t& val) {
                  val = this-> template gen_random_private<uint64_t>(); });
  std::for_each(b.data(), b.data() + b.size(),
                [this](uint64_t& val) {
                  val = this-> template gen_random_private<uint64_t>(); });
  std::for_each(alpha.data(), alpha.data() + alpha.size(),
                [this](uint64_t& val) {
                  val = this-> template gen_random_private<uint64_t>(); });

  std::vector<std::pair<uint64_t, uint64_t>> ab0;
  std::vector<std::pair<uint64_t, uint64_t>> ab1;

  std::function<std::vector<std::pair<uint64_t, uint64_t>>(size_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&)> gen_p_2arg
      = [this](size_t p, const std::vector<uint64_t>& v0, const std::vector<uint64_t>& v1) {
          return gen_product(p, v0, v1); };

  std::function<std::vector<std::pair<uint64_t, uint64_t>>(size_t, const std::vector<uint64_t>&)> gen_p_1arg
      = [this](size_t p, const std::vector<uint64_t>& v) {
          return gen_product(p, v); };

  if (party() == 0) {
      ab0 = gen_p_2arg(0, a, alpha);
      ab1 = gen_p_1arg(1, b);
  } else {
      ab0 = gen_p_1arg(0, b);
      ab1 = gen_p_2arg(1, a, alpha);
  }

  for (uint64_t i = 0; i < a.size(); i += 1) {
      std::array<int64_t, 5> item = {
          static_cast<int64_t>(a[i]),
          static_cast<int64_t>(alpha[i]),
          static_cast<int64_t>(b[i]),
          static_cast<int64_t>(signedfixed128_mult<N>(a[i], b[i]) + ab0[i].first + ab1[i].first),
          static_cast<int64_t>(signedfixed128_mult<N>(alpha[i], b[i]) + ab0[i].second + ab1[i].second)};
      _penta_triplet_buffer.push(std::move(item));
  }
}

template<typename T, size_t N>
std::vector<uint64_t> TripletGenerator<T, N>::gen_product(
                                        const std::vector<uint64_t> &input) {
  size_t word_width = 8 * sizeof(uint64_t);
  std::vector<uint64_t> ret;

  if (party() == 0) {
    std::vector<uint64_t> s1_buffer;

    std::vector<block> ot_mask;
    ot_mask.resize(input.size() * word_width);
    net()->recv(next_party(), ot_mask.data(), sizeof(block) * ot_mask.size());
    size_t ot_mask_idx = 0;
    for (const auto &a: input) {
      uint64_t ret_val = 0;

      for (uint64_t idx = 0; idx < word_width; idx += 1) {

          const block& round_ot_mask = ot_mask.at(ot_mask_idx);

          // bad naming from ot extention
          block q = ot()->ot_sender().get_ot_instance();

          q ^= (round_ot_mask & ot()->base_ot_choice());

          auto s = psi::hash_blocks({q, q ^ ot()->base_ot_choice()});
          uint64_t s0 = *reinterpret_cast<uint64_t *>(&s.first);
          uint64_t s1 = *reinterpret_cast<uint64_t *>(&s.second);

          s1 ^= lshift<N>(a, idx) - s0;

          s1_buffer.emplace_back(s1);

          ret_val += s0;
          ot_mask_idx++;
      }
      ret.emplace_back(ret_val);
    }
    net()->send(next_party(), s1_buffer.data(), sizeof(uint64_t) * s1_buffer.size());

  } else { // as ot recver

    std::vector<block> ot_masks;
    std::vector<block> t0_buffer;
    auto& ot_ext_recver = ot()->ot_receiver();
    gen_ot_masks(ot_ext_recver, input, ot_masks, t0_buffer);
    net()->send(next_party(), ot_masks.data(), sizeof(block) * ot_masks.size());
    std::vector<uint64_t> ot_msg;
    ot_msg.resize(input.size() * word_width);
    net()->recv(next_party(), ot_msg.data(), sizeof(uint64_t) * ot_msg.size());
    size_t ot_msg_idx = 0;
    uint64_t b_idx = 0;
    for (const auto &b: input) {
      uint64_t ret_val = 0;

      int b_weight = 0;

      for (size_t idx = 0; idx < word_width; idx += 1) {
        const uint64_t& round_ot_msg = ot_msg.at(ot_msg_idx);

        auto t0_hash = common::hash_block(t0_buffer[b_idx * word_width + idx]);

        uint64_t key = *reinterpret_cast<uint64_t *>(&t0_hash);

        bool b_i = (b >> idx) & 1;

        b_weight += b_i;

        ret_val +=  b_i ? round_ot_msg ^ key : -key;
        ot_msg_idx++;
      }
      // compensation for precision loss
      ret.emplace_back(ret_val + static_cast<uint64_t>(_s_fixed_point_compensation * b_weight));

      b_idx += 1;
    }
  }

  return ret;
}

template<typename T, size_t N>
std::vector<std::pair<uint64_t, uint64_t>>
TripletGenerator<T, N>::gen_product(size_t ot_sender,
                            const std::vector<uint64_t> &input0,
                            const std::vector<uint64_t> &input1) {

    size_t word_width = 8 * sizeof(uint64_t);

    std::vector<std::pair<uint64_t, uint64_t>> ret;

    if (party() == ot_sender) {

        std::vector<std::pair<uint64_t, uint64_t>> s1_buffer;

        std::vector<block> ot_mask;
        auto size = std::min(input0.size(), input1.size());
        ot_mask.resize(size * word_width);
        net()->recv(next_party(), ot_mask.data(), sizeof(block) * ot_mask.size());
        size_t ot_mask_idx = 0;
        for (auto a_iter = input0.cbegin(), alpha_iter = input1.cbegin();
             a_iter < input0.cend() && alpha_iter < input1.cend();
             ++a_iter, ++alpha_iter) {

            uint64_t ret_val[2] = {0};

            for (uint64_t idx = 0; idx < word_width; idx += 1) {

                const block& round_ot_mask = ot_mask.at(ot_mask_idx);

                // bad naming from ot extention
                block q = ot()->ot_sender().get_ot_instance();

                q ^= (round_ot_mask & ot()->base_ot_choice());

                auto s = psi::hash_blocks({q, q ^ ot()->base_ot_choice()});
                uint64_t* s0 = reinterpret_cast<uint64_t *>(&s.first);
                uint64_t* s1 = reinterpret_cast<uint64_t *>(&s.second);

                s1[0] ^= lshift<N>(*a_iter, idx) - s0[0];
                s1[1] ^= lshift<N>(*alpha_iter, idx) - s0[1];

                s1_buffer.emplace_back(std::make_pair(s1[0], s1[1]));

                ret_val[0] += s0[0];
                ret_val[1] += s0[1];
                ot_mask_idx++;
            }
            ret.emplace_back(std::make_pair(ret_val[0], ret_val[1]));
        }
        net()->send(next_party(), s1_buffer.data(), sizeof(std::pair<uint64_t, uint64_t>) * s1_buffer.size());

    } else { // as ot recver

        std::vector<block> ot_masks;
        std::vector<block> t0_buffer;
        auto& ot_ext_recver = ot()->ot_receiver();
        gen_ot_masks(ot_ext_recver, input0, ot_masks, t0_buffer);
        net()->send(next_party(), ot_masks.data(), sizeof(block) * ot_masks.size());
        std::vector<std::pair<uint64_t, uint64_t>> ot_msg;
        ot_msg.resize(input0.size() * word_width);
        net()->recv(next_party(), ot_msg.data(), sizeof(std::pair<uint64_t, uint64_t>) * ot_msg.size());
        size_t ot_msg_idx = 0;
        uint64_t b_idx = 0;
        for (const auto &b: input0) {
            uint64_t ret_val[2] = {0};

            int b_weight = 0;

            for (size_t idx = 0; idx < word_width; idx += 1) {
                const std::pair<uint64_t, uint64_t>& round_ot_msg = ot_msg.at(ot_msg_idx);

                auto t0_hash = common::hash_block(t0_buffer[b_idx * word_width + idx]);

                uint64_t* key = reinterpret_cast<uint64_t *>(&t0_hash);

                bool b_i = (b >> idx) & 1;

                b_weight += b_i;

                ret_val[0] +=  b_i ? round_ot_msg.first ^ key[0] : -key[0];
                ret_val[1] +=  b_i ? round_ot_msg.second ^ key[1] : -key[1];
                ot_msg_idx++;
            }
            // compensation for precision loss
            uint64_t loss = _s_fixed_point_compensation * b_weight;
            ret.emplace_back(std::make_pair(ret_val[0] + loss, ret_val[1] + loss));

            b_idx += 1;
        }
    }

    return ret;
}

} // namespace privc
