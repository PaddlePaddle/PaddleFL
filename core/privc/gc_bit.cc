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

#include "gc_bit.h"

namespace privc {

void garbled_and(const TensorBlock* a, const TensorBlock* b, TensorBlock* ret) {
    // refs to following paper to find this algorithm:
    // Zahur S, (2015). "Two halves make a whole"
    // following variables naming come from the paper

    auto block_shape = ret->shape();
    auto shape = block_shape;
    shape.erase(shape.begin());
    // increment index j0, j1 
    auto j0 = tensor_factory()->template create<int64_t>(shape);
    auto j1 = tensor_factory()->template create<int64_t>(shape);
    ot()->garbled_and_ctr(j0.get());
    ot()->garbled_and_ctr(j1.get());

    auto j0_ = tensor_factory()->template create<int64_t>(block_shape);
    auto j1_ = tensor_factory()->template create<int64_t>(block_shape);
    common::to_block(j0.get(), j0_.get());
    common::to_block(j1.get(), j1_.get());
    // select bit: pa, pb
    auto pa = tensor_factory()->template create<u8>(shape);
    auto pb = tensor_factory()->template create<u8>(shape);
    block_lsb(a, pa.get());
    block_lsb(b, pb.get());

    if (party() == 0) {
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
        // first half gate: wg
        auto tg = tensor_factory()->template create<int64_t>(block_shape);
        auto wg = tensor_factory()->template create<int64_t>(block_shape);
        t_first->copy(tg.get());
        tg->copy(wg.get());
        tg->bitwise_xor(t_second.get(), tg.get());

        auto mask_b = tensor_factory()->template create<int64_t>(block_shape);
        b->bitwise_xor(garbled_delta.get(), mask_b.get());
        std::pair<const TensorBlock*, const TensorBlock*> b_pair(b, mask_b.get());
        std::pair<TensorBlock*, TensorBlock*> j1_pair(j1_.get(), j1_.get());
        common::hash_blocks(b_pair, t_pair, j1_pair);
        // second half gate: we
        auto te = tensor_factory()->template create<int64_t>(block_shape);
        auto we = tensor_factory()->template create<int64_t>(block_shape);

        t_first->copy(te.get());
        te->copy(we.get());
        te->bitwise_xor(t_second.get(), te.get());
        te->bitwise_xor(a, te.get());

        auto tg_mask = tensor_factory()->template create<int64_t>(block_shape);
        auto we_mask = tensor_factory()->template create<int64_t>(block_shape);
        tg->bitwise_xor(garbled_delta.get(), tg_mask.get());
        te->bitwise_xor(a, we_mask.get());
        we->bitwise_xor(we_mask.get(), we_mask.get());
        if_then_else_plain(pb.get(), tg_mask.get(), tg.get(), tg.get());
        if_then_else_plain(pb.get(), we_mask.get(), we.get(), we.get());

        auto wg_mask = tensor_factory()->template create<int64_t>(block_shape);
        wg->bitwise_xor(tg.get(), wg_mask.get());
        if_then_else_plain(pa.get(), wg_mask.get(), wg.get(), wg.get());

        net()->send(next_party(), *tg);
        net()->send(next_party(), *te);
        // combine halves
        we->bitwise_xor(wg.get(), ret);
    } else {

        auto tg = tensor_factory()->template create<int64_t>(block_shape);
        auto te = tensor_factory()->template create<int64_t>(block_shape);
        net()->template recv(next_party(), *tg);
        net()->template recv(next_party(), *te);

        auto t_first = tensor_factory()->template create<int64_t>(block_shape);
        auto t_second = tensor_factory()->template create<int64_t>(block_shape);
        std::pair<TensorBlock*, TensorBlock*> t_pair(t_first.get(), t_second.get());
        std::pair<const TensorBlock*, const TensorBlock*> x_pair(a, b);
        std::pair<TensorBlock*, TensorBlock*> j_pair(j0_.get(), j1_.get());
        common::hash_blocks(x_pair, t_pair, j_pair);
        // first half gate: wg
        // second half gate: we
        auto wg = tensor_factory()->template create<int64_t>(block_shape);
        auto we = tensor_factory()->template create<int64_t>(block_shape);
        t_first->copy(wg.get());
        t_second->copy(we.get());
    
        auto wg_mask = tensor_factory()->template create<int64_t>(block_shape);
        wg->bitwise_xor(tg.get(), wg_mask.get());
        if_then_else_plain(pa.get(), wg_mask.get(), wg.get(), wg.get());

        auto we_mask = tensor_factory()->template create<int64_t>(block_shape);
        te->bitwise_xor(a, we_mask.get());
        we->bitwise_xor(we_mask.get(), we_mask.get());
        if_then_else_plain(pb.get(), we_mask.get(), we.get(), we.get());
        // combine halves
        wg->bitwise_xor(we.get(), ret);
    }
}

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

        auto ot_ins0 = tensor_factory()->template create<int64_t>(shape);
        auto ot_ins1 = tensor_factory()->template create<int64_t>(shape);
        ot()->ot_receiver().get_ot_instance(ot_ins0.get(), ot_ins1.get());

        auto choice = tensor_factory()->template create<int64_t>(shape);
        std::transform(val->data(), val->data() + val->numel(),
                       reinterpret_cast<block*>(choice->data()),
                       [](bool val) -> block { return val ?
                                common::OneBlock : common::ZeroBlock; });

        auto ot_mask = tensor_factory()->template create<int64_t>(shape);
        ot_ins0->bitwise_xor(ot_ins1.get(), ot_mask.get());
        ot_mask->bitwise_xor(choice.get(), ot_mask.get());

        net()->send(next_party(), *ot_mask);

        auto ot_recv = tensor_factory()->template create<int64_t>(shape);
        net()->template recv(next_party(), *ot_recv);

        auto ret_ = tensor_factory()->template create<int64_t>(shape);
        common::hash_block(ot_ins0.get(), ret_.get());

        auto ret_mask = tensor_factory()->template create<int64_t>(shape);
        ret_->bitwise_xor(ot_recv.get(), ret_mask.get());
        if_then_else_plain(val, ret_mask.get(), ret_.get(), ret_.get());
        ret_->copy(ret);
    }
}

} // namespace privc

