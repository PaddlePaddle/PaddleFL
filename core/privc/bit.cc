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

#include "bit.h"

namespace privc {

std::vector<bool> reconstruct(std::vector<Bit> bits,
              size_t party_in) {
    std::vector<u8> remote;
    std::vector<u8> local;
    std::vector<bool> ret;
    ret.resize(bits.size());
    auto party = paddle::mpc::ContextHolder::mpc_ctx()->party();
    auto next_party = paddle::mpc::ContextHolder::mpc_ctx()->next_party();
    auto net = paddle::mpc::ContextHolder::mpc_ctx()->network();
    for (auto& i : bits) {
      local.emplace_back(block_lsb(i.share()));
    }

    // make remote ^ local = 0 if party_in == next_party()
    remote = local;

    if (party_in == std::numeric_limits<size_t>::max()) {
      // reveal to all
      if (party == 0) {
        net->recv(next_party, remote.data(), remote.size() * sizeof(u8));
        net->send(next_party, local.data(), local.size() * sizeof(u8));
      } else {
        net->send(next_party, local.data(), local.size() * sizeof(u8));
        net->recv(next_party, remote.data(), remote.size() * sizeof(u8));
      }
    } else {
      //reveal to one
      if (party == party_in) {
        net->recv(next_party, remote.data(), remote.size() * sizeof(u8));
      } else {
        net->send(next_party, local.data(), local.size() * sizeof(u8));
      }
    }

    std::transform(local.begin(), local.end(),
                   remote.begin(), ret.begin(),
                   [] (u8& lhs, u8& rhs){ return lhs ^ rhs; });

    return ret;
}

block garbled_and(block a, block b) {
    auto& garbled_and_ctr = ot()->garbled_and_ctr();
    if (party() == 0) {
        u8 pa = block_lsb(a);
        u8 pb = block_lsb(b);

        u64 j0 = garbled_and_ctr += 2;
        u64 j1 = j0 + 1;

        auto j0_ = psi::to_block(j0);
        auto j1_ = psi::to_block(j1);
        auto& garbled_delta = ot()->garbled_delta();
        auto t = psi::hash_blocks({a, a ^ garbled_delta}, {j0_, j0_});

        block tg = t.first;
        block wg = tg;

        tg ^= t.second;

        t = psi::hash_blocks({b, b ^ garbled_delta}, {j1_, j1_});

        block te = t.first;
        block we = te;

        te ^= t.second;

        te ^= a;

        if (pb) {
            tg ^= garbled_delta;
            we ^= te ^ a;
        }

        if (pa) {
            wg ^= tg;
        }
        /* TODO: add gc delay
        if (_gc_delay) {
            send_to_buffer(tg);
            send_to_buffer(te);
        } else {
            send_val(tg);
            send_val(te);
        }
        */
        net()->send(next_party(), tg);
        net()->send(next_party(), te);

        return we ^ wg;
    } else {
        u8 sa = block_lsb(a);
        u8 sb = block_lsb(b);

        u64 j0 = garbled_and_ctr += 2;
        u64 j1 = j0 + 1;

        auto j0_ = psi::to_block(j0);
        auto j1_ = psi::to_block(j1);

        block tg = net()->template recv<block>(next_party());
        block te = net()->template recv<block>(next_party());

        auto t = psi::hash_blocks({a, b}, {j0_, j1_});
        block wg = t.first;
        block we = t.second;

        if (sa) {
            wg ^= tg;
        }
        if (sb) {
            we ^= te ^ a;
        }
        return wg ^ we;
    }
}

block garbled_share(bool val) {

    if (party() == 0) {
        block ot_mask = net()->template recv<block>(next_party());
        block q = ot()->ot_sender().get_ot_instance();
        q ^= ot_mask & ot()->base_ot_choice();

        auto ret_ = psi::hash_blocks({q, q ^ ot()->base_ot_choice()});
        auto& garbled_delta = ot()->garbled_delta();
        block to_send =
            ret_.first ^ ret_.second ^ garbled_delta;
        net()->send(next_party(), to_send);

        return ret_.first;
    } else {
        auto ot_ins = ot()->ot_receiver().get_ot_instance();

        block choice = val ? psi::OneBlock : psi::ZeroBlock;
        block ot_mask = ot_ins[0] ^ ot_ins[1] ^ choice;
        net()->send(next_party(), ot_mask);

        block ot_recv = net()->template recv<block>(next_party());
        block ret = psi::hash_block(ot_ins[0]);

        if (val) {
            ret ^= ot_recv;
        }
        return ret;
    }
}

std::vector<block> garbled_share_internal(const int64_t* val, size_t size) {
    std::vector<block> ret(sizeof(int64_t) * 8 * size); // 8 bit for 1 byte
    std::vector<block> send_buffer;
    std::vector<block> recv_buffer;
    recv_buffer.resize(sizeof(int64_t) * 8 * size);
    if (party() == 0) {
        net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(recv_buffer));
        for (size_t idx = 0; idx < 8 * sizeof(int64_t) * size; ++idx) {
            const block& ot_mask = recv_buffer.at(idx);
            block q = ot()->ot_sender().get_ot_instance();
            q ^= ot_mask & ot()->base_ot_choice();

            auto ret_ = psi::hash_blocks({q, q ^ ot()->base_ot_choice()});
            ret[idx] = ret_.first;
            auto& garbled_delta = ot()->garbled_delta();
            block to_send =
                ret_.second ^ ret[idx] ^ garbled_delta;
            //send_to_buffer(to_send);
            send_buffer.emplace_back(to_send);
        }

        //flush_buffer();
        net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(block));

        return ret;

    } else {

        for (size_t idx = 0; idx < 8 * sizeof(int64_t) * size; ++idx) {
            auto ot_ins = ot()->ot_receiver().get_ot_instance();
            ret[idx] = psi::hash_block(ot_ins[0]);

            size_t i = idx / (sizeof(int64_t) * 8);
            size_t j = idx % (sizeof(int64_t) * 8);

            block choice = (val[i] >> j) & 1 ? psi::OneBlock : psi::ZeroBlock;
            block ot_mask = ot_ins[0] ^ ot_ins[1] ^ choice;
            //send_to_buffer(ot_mask);
            send_buffer.emplace_back(ot_mask);
        }

        //flush_buffer();
        net()->send(next_party(), send_buffer.data(), send_buffer.size() * sizeof(block));
        net()->recv(next_party(), recv_buffer.data(), recv_buffer.size() * sizeof(recv_buffer));
        for (size_t idx = 0; idx < 8 * sizeof(int64_t) * size; ++idx) {
            const block& ot_recv = recv_buffer.at(idx);

            size_t i = idx / (sizeof(int64_t) * 8);
            size_t j = idx % (sizeof(int64_t) * 8);

            ret[idx] ^= (val[i] >> j) & 1 ? ot_recv : psi::ZeroBlock;
        }
        return ret;
    }
}

std::vector<block> garbled_share(int64_t val) {
    return garbled_share_internal(&val, 1);
}

std::vector<block> garbled_share(const std::vector<int64_t>& val) {
    return garbled_share_internal(val.data(), val.size());
}

} // namespace privc

