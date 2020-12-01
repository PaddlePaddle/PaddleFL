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

/*std::vector<bool> reconstruct(std::vector<Bit> bits,
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
}*/

void garbled_and(const TensorBlock* a, const TensorBlock* b, TensorBlock* ret) {
    //auto& garbled_and_ctr = ot()->garbled_and_ctr();
    auto block_shape = ret->shape();
    auto shape = block_shape;
    shape.erase(shape.begin());
    auto j0 = tensor_factory()->template create<int64_t>(shape);
    auto j1 = tensor_factory()->template create<int64_t>(shape);
    ot()->garbled_and_ctr(j0.get());
    ot()->garbled_and_ctr(j1.get());

    auto j0_ = tensor_factory()->template create<int64_t>(block_shape);
    auto j1_ = tensor_factory()->template create<int64_t>(block_shape);
    common::to_block(j0.get(), j0_.get());
    common::to_block(j1.get(), j1_.get());

    auto pa = tensor_factory()->template create<u8>(shape);
    auto pb = tensor_factory()->template create<u8>(shape);
    block_lsb(a, pa.get());
    block_lsb(b, pb.get());

    if (party() == 0) {
        //u8 pa = block_lsb(a);
        //u8 pb = block_lsb(b);


        //u64 j0 = garbled_and_ctr += 2;
        //u64 j1 = j0 + 1;

        //auto j0_ = common::to_block(j0);
        //auto j1_ = common::to_block(j1);
        //auto& garbled_delta = ot()->garbled_delta();
        //auto t = common::hash_blocks({a, a ^ garbled_delta}, {j0_, j0_});
        auto garbled_delta = tensor_factory()->template create<int64_t>(block_shape);
        ot()->garbled_delta(garbled_delta.get());
        auto mask_a = tensor_factory()->template create<int64_t>(block_shape);
        a->bitwise_xor(garbled_delta.get(), mask_a.get());
        std::pair<const TensorBlock*, const TensorBlock*> a_pair(a, mask_a.get());
        std::pair<TensorBlock*, TensorBlock*> j0_pair(j0_.get(), j0_.get());
        auto t_first = tensor_factory()->template create<int64_t>(block_shape);
        auto t_second = tensor_factory()->template create<int64_t>(block_shape);
        std::pair<TensorBlock*, TensorBlock*> t_pair(t_first.get(), t_second.get());
        common::hash_blocks(a_pair, t_pair, j0_pair);

        //block tg = t.first;
        //block wg = tg;

        //tg ^= t.second;
        auto tg = tensor_factory()->template create<int64_t>(block_shape);
        auto wg = tensor_factory()->template create<int64_t>(block_shape);
        t_first->copy(tg.get());
        tg->copy(wg.get());
        tg->bitwise_xor(t_second.get(), tg.get());

        //t = common::hash_blocks({b, b ^ garbled_delta}, {j1_, j1_});
        auto mask_b = tensor_factory()->template create<int64_t>(block_shape);
        b->bitwise_xor(garbled_delta.get(), mask_b.get());
        std::pair<const TensorBlock*, const TensorBlock*> b_pair(b, mask_b.get());
        std::pair<TensorBlock*, TensorBlock*> j1_pair(j1_.get(), j1_.get());
        common::hash_blocks(b_pair, t_pair, j1_pair);

        //block te = t.first;
        //block we = te;

        //te ^= t.second;

        //te ^= a;
        auto te = tensor_factory()->template create<int64_t>(block_shape);
        auto we = tensor_factory()->template create<int64_t>(block_shape);

        t_first->copy(te.get());
        te->copy(we.get());
        te->bitwise_xor(t_second.get(), te.get());
        te->bitwise_xor(a, te.get());

        //if (pb) {
        //    tg ^= garbled_delta;
        //    we ^= te ^ a;
        //}
        auto tg_mask = tensor_factory()->template create<int64_t>(block_shape);
        auto we_mask = tensor_factory()->template create<int64_t>(block_shape);
        tg->bitwise_xor(garbled_delta.get(), tg_mask.get());
        te->bitwise_xor(a, we_mask.get());
        we->bitwise_xor(we_mask.get(), we_mask.get());
        if_then_else_plain(pb.get(), tg_mask.get(), tg.get(), tg.get());
        if_then_else_plain(pb.get(), we_mask.get(), we.get(), we.get());

        //if (pa) {
        //    wg ^= tg;
        //}
        auto wg_mask = tensor_factory()->template create<int64_t>(block_shape);
        wg->bitwise_xor(tg.get(), wg_mask.get());
        if_then_else_plain(pa.get(), wg_mask.get(), wg.get(), wg.get());
        /* TODO: add gc delay
        if (_gc_delay) {
            send_to_buffer(tg);
            send_to_buffer(te);
        } else {
            send_val(tg);
            send_val(te);
        }
        */
        net()->send(next_party(), *tg);
        net()->send(next_party(), *te);

        //return we ^ wg;
        we->bitwise_xor(wg.get(), ret);
    } else {
        //u8 sa = block_lsb(a);
        //u8 sb = block_lsb(b);

        //u64 j0 = garbled_and_ctr += 2;
        //u64 j1 = j0 + 1;

        //auto j0_ = common::to_block(j0);
        //auto j1_ = common::to_block(j1);
        auto tg = tensor_factory()->template create<int64_t>(block_shape);
        auto te = tensor_factory()->template create<int64_t>(block_shape);
        net()->template recv(next_party(), *tg);
        net()->template recv(next_party(), *te);

        //auto t = common::hash_blocks({a, b}, {j0_, j1_});
        //block wg = t.first;
        //block we = t.second;
        auto t_first = tensor_factory()->template create<int64_t>(block_shape);
        auto t_second = tensor_factory()->template create<int64_t>(block_shape);
        std::pair<TensorBlock*, TensorBlock*> t_pair(t_first.get(), t_second.get());
        std::pair<const TensorBlock*, const TensorBlock*> x_pair(a, b);
        std::pair<TensorBlock*, TensorBlock*> j_pair(j0_.get(), j1_.get());
        common::hash_blocks(x_pair, t_pair, j_pair);

        auto wg = tensor_factory()->template create<int64_t>(block_shape);
        auto we = tensor_factory()->template create<int64_t>(block_shape);
        t_first->copy(wg.get());
        t_second->copy(we.get());
        

        //if (sa) {
        //    wg ^= tg;
        //}
        auto wg_mask = tensor_factory()->template create<int64_t>(block_shape);
        wg->bitwise_xor(tg.get(), wg_mask.get());
        if_then_else_plain(pa.get(), wg_mask.get(), wg.get(), wg.get());

        //if (sb) {
        //    we ^= te ^ a;
        //}
        auto we_mask = tensor_factory()->template create<int64_t>(block_shape);
        te->bitwise_xor(a, we_mask.get());
        we->bitwise_xor(we_mask.get(), we_mask.get());
        if_then_else_plain(pb.get(), we_mask.get(), we.get(), we.get());

        //return wg ^ we;
        wg->bitwise_xor(we.get(), ret);
    }
}

/*block garbled_and(block a, block b) {
    auto& garbled_and_ctr = ot()->garbled_and_ctr();
    if (party() == 0) {
        u8 pa = block_lsb(a);
        u8 pb = block_lsb(b);

        u64 j0 = garbled_and_ctr += 2;
        u64 j1 = j0 + 1;

        auto j0_ = common::to_block(j0);
        auto j1_ = common::to_block(j1);
        auto& garbled_delta = ot()->garbled_delta();
        auto t = common::hash_blocks({a, a ^ garbled_delta}, {j0_, j0_});

        block tg = t.first;
        block wg = tg;

        tg ^= t.second;

        t = common::hash_blocks({b, b ^ garbled_delta}, {j1_, j1_});

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
        /*net()->send(next_party(), tg);
        net()->send(next_party(), te);

        return we ^ wg;
    } else {
        u8 sa = block_lsb(a);
        u8 sb = block_lsb(b);

        u64 j0 = garbled_and_ctr += 2;
        u64 j1 = j0 + 1;

        auto j0_ = common::to_block(j0);
        auto j1_ = common::to_block(j1);

        block tg = net()->template recv<block>(next_party());
        block te = net()->template recv<block>(next_party());

        auto t = common::hash_blocks({a, b}, {j0_, j1_});
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
*/
void garbled_share(const TensorAdapter<u8>* val, TensorBlock* ret) {
    auto shape = ret->shape();
    if (party() == 0) {
        auto ot_mask = tensor_factory()->template create<int64_t>(shape);
        net()->template recv(next_party(), *ot_mask);
        auto q = tensor_factory()->template create<int64_t>(shape);
        auto base_ot_choice = tensor_factory()->template create<int64_t>(shape);
        ot()->base_ot_choice(base_ot_choice.get());
        ot()->ot_sender().get_ot_instance(q.get());
        ot_mask->bitwise_and(base_ot_choice.get(), ot_mask.get());
        q->bitwise_xor(ot_mask.get(), q.get());
        //q ^= ot_mask & ot()->base_ot_choice();

        // auto ret_ = common::hash_blocks({q, q ^ ot()->base_ot_choice()});
        // auto& garbled_delta = ot()->garbled_delta();
        // block to_send =
        //     ret_.first ^ ret_.second ^ garbled_delta;

        auto q_mask = tensor_factory()->template create<int64_t>(shape);
        q->bitwise_xor(base_ot_choice.get(), q_mask.get());
        std::pair<const TensorBlock*, const TensorBlock*> q_pair(q.get(), q_mask.get());
        auto* ret_first = ret;
        auto ret_second = tensor_factory()->template create<int64_t>(shape);
        std::pair<TensorBlock*, TensorBlock*> ret_pair(ret_first, ret_second.get());
        common::hash_blocks(q_pair, ret_pair);

        auto to_send = tensor_factory()->template create<int64_t>(shape);
        ret_first->bitwise_xor(ret_second.get(), to_send.get());
        auto garbled_delta = tensor_factory()->template create<int64_t>(shape);
        ot()->garbled_delta(garbled_delta.get());
        to_send->bitwise_xor(garbled_delta.get(), to_send.get());
        
        net()->send(next_party(), *to_send);

    } else {
        //auto ot_ins = ot()->ot_receiver().get_ot_instance();
        auto ot_ins0 = tensor_factory()->template create<int64_t>(shape);
        auto ot_ins1 = tensor_factory()->template create<int64_t>(shape);
        ot()->ot_receiver().get_ot_instance(ot_ins0.get(), ot_ins1.get());

        //block choice = val ? common::OneBlock : ZeroBlock;
        auto choice = tensor_factory()->template create<int64_t>(shape);
        std::transform(val->data(), val->data() + val->numel(),
                       reinterpret_cast<block*>(choice->data()),
                       [](bool val) -> block { return val ? common::OneBlock : common::ZeroBlock; });

        //block ot_mask = ot_ins[0] ^ ot_ins[1] ^ choice;
        auto ot_mask = tensor_factory()->template create<int64_t>(shape);
        ot_ins0->bitwise_xor(ot_ins1.get(), ot_mask.get());
        ot_mask->bitwise_xor(choice.get(), ot_mask.get());

        net()->send(next_party(), *ot_mask);

        auto ot_recv = tensor_factory()->template create<int64_t>(shape);
        net()->template recv(next_party(), *ot_recv);
        //block ret = common::hash_block(ot_ins[0]);
        auto ret_ = tensor_factory()->template create<int64_t>(shape);
        common::hash_block(ot_ins0.get(), ret_.get());

        //if (val) {
        //    ret ^= ot_recv;
        //}

        auto ret_mask = tensor_factory()->template create<int64_t>(shape);
        ret_->bitwise_xor(ot_recv.get(), ret_mask.get());
        if_then_else_plain(val, ret_mask.get(), ret_.get(), ret_.get());
        ret_->copy(ret);
        //return ret;
    }
}

/*block garbled_share(bool val) {

    if (party() == 0) {
        block ot_mask = net()->template recv<block>(next_party());
        block q = ot()->ot_sender().get_ot_instance();
        q ^= ot_mask & ot()->base_ot_choice();

        auto ret_ = common::hash_blocks({q, q ^ ot()->base_ot_choice()});
        auto& garbled_delta = ot()->garbled_delta();
        block to_send =
            ret_.first ^ ret_.second ^ garbled_delta;
        net()->send(next_party(), to_send);

        return ret_.first;
    } else {
        auto ot_ins = ot()->ot_receiver().get_ot_instance();

        block choice = val ? common::OneBlock : common::ZeroBlock;
        block ot_mask = ot_ins[0] ^ ot_ins[1] ^ choice;
        net()->send(next_party(), ot_mask);

        block ot_recv = net()->template recv<block>(next_party());
        block ret = common::hash_block(ot_ins[0]);

        if (val) {
            ret ^= ot_recv;
        }
        return ret;
    }
}*/
/*
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

            auto ret_ = common::hash_blocks({q, q ^ ot()->base_ot_choice()});
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
            ret[idx] = common::hash_block(ot_ins[0]);

            size_t i = idx / (sizeof(int64_t) * 8);
            size_t j = idx % (sizeof(int64_t) * 8);

            block choice = (val[i] >> j) & 1 ? common::OneBlock : common::ZeroBlock;
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

            ret[idx] ^= (val[i] >> j) & 1 ? ot_recv : common::ZeroBlock;
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
*/
} // namespace privc

