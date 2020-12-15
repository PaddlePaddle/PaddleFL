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
#include "../common/prng.h"
#include "core/common/paddle_tensor.h"

namespace privc {

template<typename T>
inline void if_then_else_plain(const TensorAdapter<T>* val,
                         const TensorBlock* then_val,
                         const TensorBlock* else_val,
                         TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(_g_block_size_expand * val->numel(),
                      then_val->numel(), "input of then val's numel no match with input val.");
    PADDLE_ENFORCE_EQ(else_val->numel(), then_val->numel(),
                      "input of else val's numel no match.");
    PADDLE_ENFORCE_EQ(ret->numel(), then_val->numel(),
                      "input of then val's numel no match with return.");

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


inline void garbled_and(const TensorBlock* a, const TensorBlock* b, TensorBlock* ret) {
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
    // convert int64 to block
    common::to_block(j0.get(), j0_.get());
    common::to_block(j1.get(), j1_.get());
    // select bit: pa, pb
    auto pa = tensor_factory()->template create<u8>(shape);
    auto pb = tensor_factory()->template create<u8>(shape);
    block_lsb(a, pa.get());
    block_lsb(b, pb.get());

    if (party() == 0) {
        // get garbled delta : R
        auto R = tensor_factory()->template create<int64_t>(block_shape);
        ot()->garbled_delta(R.get());

        auto mask_a = tensor_factory()->template create<int64_t>(block_shape);
        a->bitwise_xor(R.get(), mask_a.get());

        // cal t = H(a, j), H(mask_a, j)
        std::pair<const TensorBlock*, const TensorBlock*> a_pair(a, mask_a.get());
        std::pair<TensorBlock*, TensorBlock*> j0_pair(j0_.get(), j0_.get());
        auto t_first = tensor_factory()->template create<int64_t>(block_shape);
        auto t_second = tensor_factory()->template create<int64_t>(block_shape);
        std::pair<TensorBlock*, TensorBlock*> t_pair(t_first.get(), t_second.get());
        common::hash_blocks(a_pair, t_pair, j0_pair);

        // tg = H(a, j) ^ H(mask_a, j) ^ pa*R
        auto tg = tensor_factory()->template create<int64_t>(block_shape);

        // first half gate: wg = H(a, j) ^ pa*tg
        auto wg = tensor_factory()->template create<int64_t>(block_shape);
        t_first->copy(tg.get());
        tg->copy(wg.get());
        tg->bitwise_xor(t_second.get(), tg.get());

        auto mask_b = tensor_factory()->template create<int64_t>(block_shape);
        b->bitwise_xor(R.get(), mask_b.get());
        std::pair<const TensorBlock*, const TensorBlock*> b_pair(b, mask_b.get());
        std::pair<TensorBlock*, TensorBlock*> j1_pair(j1_.get(), j1_.get());
        common::hash_blocks(b_pair, t_pair, j1_pair);

        // te = H(b, j1) ^ H(mask_b, j1) ^ a
        auto te = tensor_factory()->template create<int64_t>(block_shape);
        // second half gate: we = H(b, j1) ^ pb(te ^ a)
        auto we = tensor_factory()->template create<int64_t>(block_shape);

        t_first->copy(te.get());
        te->copy(we.get());
        te->bitwise_xor(t_second.get(), te.get());
        te->bitwise_xor(a, te.get());

        auto tg_mask = tensor_factory()->template create<int64_t>(block_shape);
        auto we_mask = tensor_factory()->template create<int64_t>(block_shape);
        tg->bitwise_xor(R.get(), tg_mask.get());
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
        // recv tg, te
        auto tg = tensor_factory()->template create<int64_t>(block_shape);
        auto te = tensor_factory()->template create<int64_t>(block_shape);
        net()->template recv(next_party(), *tg);
        net()->template recv(next_party(), *te);
        // cal t = H(a, j0), H(b, j1)
        auto t_first = tensor_factory()->template create<int64_t>(block_shape);
        auto t_second = tensor_factory()->template create<int64_t>(block_shape);
        std::pair<TensorBlock*, TensorBlock*> t_pair(t_first.get(), t_second.get());
        std::pair<const TensorBlock*, const TensorBlock*> x_pair(a, b);
        std::pair<TensorBlock*, TensorBlock*> j_pair(j0_.get(), j1_.get());
        common::hash_blocks(x_pair, t_pair, j_pair);
        // first half gate: wg = H(a, j0) ^ pa*tg
        auto wg = tensor_factory()->template create<int64_t>(block_shape);
        // second half gate: we = H(b, j1) ^ pb*(te ^ a)
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

inline void garbled_share(const TensorAdapter<u8>* val, TensorBlock* ret) {
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

inline void to_gc_bit(TensorAdapter<u8>* val, size_t party_in, TensorBlock* ret) {
    auto block_shape = val->shape();
    block_shape.insert(block_shape.begin(), 2);
    if (party_in == 0) {
        if (party() == 0) {
            privc_ctx()->template gen_random_private(*ret);

            auto to_send = tensor_factory()->template create<int64_t>(block_shape);
            ret->copy(to_send.get());
            auto mask_to_send = tensor_factory()->template create<int64_t>(block_shape);
            auto garbled_delta = tensor_factory()->template create<int64_t>(block_shape);
            ot()->garbled_delta(garbled_delta.get());
            to_send->bitwise_xor(garbled_delta.get(), mask_to_send.get());
            if_then_else_plain(val, mask_to_send.get(), to_send.get(), to_send.get());

            net()->send(next_party(), *to_send);

        } else {
            // maybe need to know recv shape
            net()->recv(next_party(), *ret);
        }
    } else {
        garbled_share(val, ret);
    }
}


inline std::shared_ptr<TensorBlock> create_gc_share(const std::vector<size_t>& gc_shape) {
    auto ret = tensor_factory()->template create<int64_t>(gc_shape);
    std::for_each(ret->data(), ret->data() + ret->numel(),
                  [](int64_t& a){ a = 0; });
    return ret;
}

inline void garbled_or(const TensorBlock* lhs, 
                const TensorBlock* rhs, TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(rhs->numel(), ret->numel(),
                        "input of rhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(ret->numel(), lhs->numel(),
                        "input of lhs's numel no match with return.");

    auto op_t = create_gc_share(lhs->shape());
    garbled_and(lhs, rhs, op_t.get());
    lhs->bitwise_xor(op_t.get(), op_t.get());
    rhs->bitwise_xor(op_t.get(), ret);
}

inline void garbled_not(const TensorBlock* lhs, TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(ret->numel(), lhs->numel(),
                        "input numel no match.");

    lhs->copy(ret);
    auto garbled_delta = tensor_factory()->template create<int64_t>(lhs->shape());
    ot()->garbled_delta(garbled_delta.get());
    if (party() == 0) {
        ret->bitwise_xor(garbled_delta.get(), ret);
    }
}



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

inline void to_gc_num(double in, TensorBlock* gc_share, size_t N) {
    auto gc_shape = gc_share->shape();
    int length = gc_shape[0];
    std::for_each(gc_share->data(), gc_share->data() + gc_share->numel(),
                    [](int64_t& a) { a = 0; });
    int64_t in_ = (int64_t) (in * std::pow(2, N));
    for (int i = 0; i < length; i += 1) {
        if (party() == 0 && in_ >> i & 1) {
            auto share_i = (*gc_share)[i];
            auto garbled_delta = tensor_factory()->template create<int64_t>(share_i->shape());
            ot()->garbled_delta(garbled_delta.get());
            garbled_delta->copy(share_i.get());
        }
    }
}

inline void lsb(const TensorBlock* lhs, TensorAdapter<int64_t>* ret) {
    // get least significate bit
    PADDLE_ENFORCE_EQ(lhs->numel() / lhs->shape()[0] / lhs->shape()[1],
                    ret->numel(), "input numel no match.");

    std::for_each(ret->data(), ret->data() + ret->numel(),
                    [](int64_t& a) { a = 0;});
    for (int idx = 0; idx < lhs->shape()[0]; idx += 1) {
        auto tmp = tensor_factory()->template create<int64_t>(ret->shape());
        block_lsb((*lhs)[idx].get(), tmp.get());

        tmp->lshift(idx, tmp.get());

        ret->bitwise_or(tmp.get(), ret);
    }
}


inline void if_then_else(TensorBlock* dest, const TensorBlock *tsrc,
                  const TensorBlock *fsrc, int size,
                  TensorBlock* cond, int pos_dest = 0,
                  int pos_tsrc = 0, int pos_fsrc = 0) {
    int i = 0;
    while (size-- > 0) {
        // dest[i] = cond[i] & (t[i] ^ f[i]) ^ f[i]
        auto x = create_gc_share((*tsrc)[i + pos_tsrc]->shape());
        (*tsrc)[i + pos_tsrc]->bitwise_xor((*fsrc)[i + pos_fsrc].get(), x.get());

        auto a = create_gc_share((*tsrc)[i + pos_tsrc]->shape());

        garbled_and(cond, x.get(), a.get());
        a->bitwise_xor((*fsrc)[i + pos_fsrc].get(), (*dest)[i + pos_dest].get());
        ++i;
    }
}


inline void if_then_else(TensorBlock* cond, const TensorBlock* t_int,
                              const TensorBlock* f_int,
                              TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(cond->numel() * sizeof(int64_t) * 8,
                      ret->numel(),
                      "input of condition's numel no match with return.");
    PADDLE_ENFORCE_EQ(t_int->numel(),
                      ret->numel(),
                      "input of true val's numel no match with return.");
    PADDLE_ENFORCE_EQ(f_int->numel(),
                      ret->numel(),
                      "input of false val's numel no match with return.");

    auto res = create_gc_share(t_int->shape());
    privc::if_then_else(res.get(), t_int, f_int, res->shape()[0], cond);

    res->copy(ret);
}

inline void add_full(TensorBlock *dest, TensorBlock *carry_out,
              const TensorBlock *op1, const TensorBlock *op2,
              const TensorBlock *carry_in, int size, size_t pos_dest = 0,
              size_t pos_op1 = 0, size_t pos_op2 = 0) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());

    auto carry = create_gc_share(bit_shape);
    auto bxc = create_gc_share(bit_shape);
    auto axc = create_gc_share(bit_shape);
    auto t = create_gc_share(bit_shape);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (carry_in && carry_out) {
            carry_in->copy(carry_out);
        }
        return;
    }
    if (carry_in) {
        carry_in->copy(carry.get());
    }
    // skip AND on last bit if carry_out==NULL
    skip_last = (carry_out == nullptr);
    while (size-- > skip_last) {
        (*op1)[i + pos_op1]->bitwise_xor(carry.get(), axc.get());
        (*op2)[i + pos_op2]->bitwise_xor(carry.get(), bxc.get());
        (*op1)[i + pos_op1]->bitwise_xor(bxc.get(), (*dest)[i + pos_dest].get());

        garbled_and(axc.get(), bxc.get(), t.get());
        carry->bitwise_xor(t.get(), carry.get());
        ++i;
    }
    if (carry_out != nullptr) {
        carry->copy(carry_out);
    } else {
        carry->bitwise_xor((*op2)[i + pos_op2].get(), (*dest)[i + pos_dest].get());
        (*dest)[i + pos_dest]->
            bitwise_xor((*op1)[i + pos_op1].get(), (*dest)[i + pos_dest].get());
    }
        
}

inline void sub_full(TensorBlock *dest, TensorBlock *borrow_out,
              const TensorBlock *op1, const TensorBlock *op2,
              const TensorBlock *borrow_in, int size, int pos_dest = 0,
              int pos_op1 = 0, int pos_op2 = 0) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());

    auto borrow = create_gc_share(bit_shape);
    auto bxc = create_gc_share(bit_shape);
    auto bxa = create_gc_share(bit_shape);
    auto t = create_gc_share(bit_shape);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (borrow_in && borrow_out) {
            //borrow_out = borrow_in;
            borrow_in->copy(borrow_out);
        }
        return;
    }
    if (borrow_in) {
        // borrow = borrow_in;
        borrow_in->copy(borrow.get());
    }
    // skip AND on last bit if borrow_out==NULL
    skip_last = (borrow_out == nullptr);
    while (size-- > skip_last) {
        (*op1)[i + pos_op1]->bitwise_xor((*op2)[i + pos_op2].get(), bxa.get());
        borrow->bitwise_xor((*op2)[i + pos_op2].get(), bxc.get());
        bxa->bitwise_xor(borrow.get(), (*dest)[i + pos_dest].get());

        garbled_and(bxa.get(), bxc.get(), t.get());
        borrow->bitwise_xor(t.get(), borrow.get());
        ++i;
    }
    if (borrow_out != nullptr) {
        // borrow_out = borrow;
        borrow->copy(borrow_out);
    } else {
        (*op1)[i + pos_op1]->bitwise_xor((*op2)[i + pos_op2].get(), (*dest)[i + pos_dest].get());
        (*dest)[i + pos_dest]->bitwise_xor(borrow.get(), (*dest)[i + pos_dest].get());
    }
}

inline void mul_full(TensorBlock *dest, const TensorBlock *op1,
              const TensorBlock *op2, int size) {
    auto sum = create_gc_share(dest->shape());
    auto tmp = create_gc_share(dest->shape());

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size - i; ++k) {
            garbled_and((*op1)[k].get(), (*op2)[i].get(), (*tmp)[k].get());
        }
        add_full(sum.get(), nullptr, sum.get(), tmp.get(), nullptr, size - i, i, i, 0);
    }
    // calc dest sign
    (*op1)[size - 1]->bitwise_xor((*op2)[size - 1].get(), (*sum)[size - 1].get());
    sum->copy(dest);
}

inline void cond_neg(TensorBlock* cond, TensorBlock *dest,
              const TensorBlock *src) {
    int i = 0;
    auto c = create_gc_share(cond->shape());
    cond->copy(c.get());
    for (i = 0; i < dest->shape()[0] - 1; ++i) {

        (*src)[i]->bitwise_xor(cond, (*dest)[i].get());

        auto t = create_gc_share(c->shape());
        (*dest)[i]->bitwise_xor(c.get(), t.get());

        garbled_and(c.get(), (*dest)[i].get(), c.get());

        t->copy((*dest)[i].get());
    }

    c->bitwise_xor(cond, c.get());
    c->bitwise_xor((*src)[i].get(), (*dest)[i].get());
}

inline void div_full(TensorBlock*vquot, TensorBlock *vrem,
              const TensorBlock *op1, const TensorBlock *op2) {

    PADDLE_ENFORCE_EQ(op1->numel(), op2->numel(),
                      "input numel no match");
    auto shape = op1->shape();
    auto size = shape[0];

    auto overflow = create_gc_share(shape);
    auto tmp = create_gc_share(shape);
    auto rem = create_gc_share(shape);
    auto quot = create_gc_share(shape);

    auto bit_shape = shape;
    bit_shape.erase(bit_shape.begin());

    auto b = create_gc_share(bit_shape);
    op1->copy(rem.get());

    for (int i = 1; i < size; ++i) {
        garbled_or((*overflow)[i - 1].get(), (*op2)[size - i].get(), (*overflow)[i].get());
    }
    // skip AND on last bit if borrow_out==NULL
    for (int i = size - 1; i >= 0; --i) {
        sub_full(tmp.get(), b.get(), rem.get(), op2, nullptr, size - i, 0, i, 0);
        garbled_or(b.get(), (*overflow)[i].get(), b.get());
        if_then_else(rem.get(), rem.get(), tmp.get(), size - i, b.get(), i, i, 0);
        garbled_not(b.get(), (*quot)[i].get());
    }
    if (vrem != nullptr) {
        // vrem = rem
        rem->copy(vrem);
    }
    if (vquot != nullptr) {
        // vquot = quot
        quot->copy(vquot);
    }
}


template<typename T, size_t N>
void FixedPointTensor<T, N>::to_gc_num(const TensorAdapter<int64_t>* input, size_t party_in,
                                       TensorBlock* gc_share) {
    // construct gc integer from ac input
    int length = sizeof(int64_t) * 8; // 1 byte = 8 bits

    auto shape = input->shape();
    auto bit_shape = shape;
    bit_shape.insert(bit_shape.begin(), length);

    auto gc_shape = bit_shape;
    gc_shape.insert(gc_shape.begin() + 1, _g_block_size_expand);

    auto input_bits = tensor_factory()->template create<u8>(bit_shape);
    // expand input to bit tensor
    to_bits(input, input_bits.get());

    if (party_in == 0) {
        if (party() == 0) {
            auto to_send = tensor_factory()->template create<int64_t>(gc_shape);
            privc_ctx()->template gen_random_private(*to_send);
            to_send->copy(gc_share);

            auto mask_val = tensor_factory()->template create<int64_t>(gc_shape);
            auto garbled_delta = tensor_factory()->template create<int64_t>(gc_shape);
            auto zero_block_tensor = tensor_factory()->template create<int64_t>(gc_shape);
            std::for_each(zero_block_tensor->data(),
                          zero_block_tensor->data() + zero_block_tensor->numel(),
                          [](int64_t& a) { a = 0; });
            ot()->garbled_delta(garbled_delta.get());
            if_then_else_plain(input_bits.get(), garbled_delta.get(),
                               zero_block_tensor.get(), mask_val.get());

            to_send->bitwise_xor(mask_val.get(), to_send.get());
            net()->send(next_party(), *to_send);
        } else {
            net()->recv(next_party(), *gc_share);
        }
    } else {
        garbled_share(input_bits.get(), gc_share);
    }
}

// Comparisons
inline void geq(const TensorBlock* lhs, const TensorBlock *rhs, TensorBlock* ret) {
    auto shape = lhs->shape();
    int size = shape[0];
    PADDLE_ENFORCE_EQ(lhs->numel() / lhs->shape()[0],
                      ret->numel(),
                      "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->numel() / lhs->shape()[0],
                      ret->numel(),
                      "input of rhs's numel no match.");

    auto dest = create_gc_share(shape);
    auto bit_shape = shape;
    bit_shape.erase(bit_shape.begin());

    auto borrow_out = create_gc_share(bit_shape);

    sub_full(dest.get(), borrow_out.get(), lhs, rhs, nullptr, size);

    (*lhs)[size - 1]->bitwise_xor((*rhs)[size - 1].get(), ret);
    ret->bitwise_xor(borrow_out.get(), ret);

    garbled_not(ret, ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::abs(const TensorBlock* lhs, TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(lhs->numel(), ret->numel(),
                      "input numel no match.");

    auto shape = lhs->shape();
    auto res = create_gc_share(shape);
    int size = shape[0];
    for (int i = 0; i < size; ++i) {
        (*lhs)[size - 1]->copy((*res)[i].get());
    }

    gc_add(lhs, res.get(), ret);
    ret->bitwise_xor(res.get(), ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::gc_add(const TensorBlock* lhs, const TensorBlock* rhs,
                                TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(lhs->numel(), ret->numel(),
                      "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->numel(), ret->numel(),
                      "input of rhs's numel no match with return.");

    add_full(ret, nullptr, lhs, rhs, nullptr, lhs->shape()[0]);
}

inline void bc_mux(const TensorAdapter<u8>* choice,
            const TensorAdapter<int64_t>* val_t,
            const TensorAdapter<int64_t>* val_f,
            TensorAdapter<int64_t>* ret) {
    PADDLE_ENFORCE_EQ(choice->numel(), ret->numel(),
            "input of choice's numel no match with return.");
    PADDLE_ENFORCE_EQ(val_t->numel(), ret->numel(),
            "input of true val's numel no match with return.");
    PADDLE_ENFORCE_EQ(val_f->numel(), ret->numel(),
            "input of false val's numel no match with return.");

    auto send_ot = [](const TensorAdapter<int64_t>* diff,
                          const TensorBlock* round_ot_mask,
                          TensorAdapter<int64_t>* send_buffer,
                          TensorAdapter<int64_t>* ret) {
        // bad naming from ot extention
        // send_buffer = s1 = diff ^ ot_instance0 ^ ot_instance1
        // ret = s0 = ot_instance0
        auto q = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        ot()->ot_sender().get_ot_instance(q.get());

        auto base_ot_choice = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        ot()->base_ot_choice(base_ot_choice.get());
        auto tmp = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        round_ot_mask->bitwise_and(base_ot_choice.get(), tmp.get());
        q->bitwise_xor(tmp.get(), q.get());

        auto s_first = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        auto s_second = tensor_factory()->template create<int64_t>(round_ot_mask->shape());
        q->bitwise_xor(base_ot_choice.get(), tmp.get());
        std::pair<TensorBlock*, TensorBlock*> x_pair({q.get(), tmp.get()});
        std::pair<TensorBlock*, TensorBlock*> s_pair({s_first.get(), s_second.get()});
        common::hash_blocks(x_pair, s_pair);

        auto s0 = tensor_factory()->template create<int64_t>(ret->shape());
        block_to_int64(s_first.get(), s0.get());

        auto msg1 = tensor_factory()->template create<int64_t>(s0->shape());
        diff->bitwise_xor(s0.get(), msg1.get());

        auto s1 = tensor_factory()->template create<int64_t>(send_buffer->shape());
        block_to_int64(s_second.get(), s1.get());

        s1->bitwise_xor(msg1.get(), s1.get());

        s1->copy(send_buffer);

        s0->copy(ret);
    };

    auto send_ot_mask = [](const TensorAdapter<u8>* choice, TensorBlock* send_buffer,
                           TensorAdapter<int64_t>* ret) {
        // ret = H(ot_instance0)
        auto block_shape = send_buffer->shape();
        auto ot_ins0 = tensor_factory()->template create<int64_t>(block_shape);
        auto ot_ins1 = tensor_factory()->template create<int64_t>(block_shape);
        ot()->ot_receiver().get_ot_instance(ot_ins0.get(), ot_ins1.get());

        auto choice_ = tensor_factory()->template create<int64_t>(block_shape);
        block* choice_ptr = reinterpret_cast<block*>(choice_->data());
        std::transform(choice->data(), choice->data() + choice->numel(),
                       choice_ptr, [](u8 val) {
                           return val ? common::OneBlock : common::ZeroBlock;
                        });

        const auto& t0 = ot_ins0;
        // ot_mask = choice ^ ot_instance0 ^ ot_instance1
        auto ot_mask = tensor_factory()->template create<int64_t>(block_shape);
        choice_->bitwise_xor(ot_ins0.get(), ot_mask.get());
        ot_mask->bitwise_xor(ot_ins1.get(), ot_mask.get());

        ot_mask->copy(send_buffer);

        auto t0_hash = tensor_factory()->template create<int64_t>(block_shape);
        common::hash_block(t0.get(), t0_hash.get());
        block_to_int64(t0_hash.get(), ret);
    };

    auto recv_ot = [](const TensorAdapter<u8>* choice, const TensorAdapter<int64_t>* key,
                      const TensorAdapter<int64_t>* round_ot_msg,
                      TensorAdapter<int64_t>* ret) {

        auto tmp = tensor_factory()->template create<int64_t>(key->shape());
        round_ot_msg->bitwise_xor(key, tmp.get());
        if_then_else_plain(false, choice, tmp.get(), key, ret);
    };

    auto shape = ret->shape();
    auto block_shape = get_block_shape(shape);
    auto delta = tensor_factory()->template create<int64_t>(shape);
    auto msg = tensor_factory()->template create<int64_t>(shape);
    auto key = tensor_factory()->template create<int64_t>(shape);

    auto send_buffer = tensor_factory()->template create<int64_t>(shape);
    auto send_buffer1 = tensor_factory()->template create<int64_t>(block_shape);
    auto recv_buffer = tensor_factory()->template create<int64_t>(block_shape);
    auto recv_buffer1 = tensor_factory()->template create<int64_t>(shape);

    auto ret_ = tensor_factory()->template create<int64_t>(shape);


    if (party() == 0) {
        net()->recv(next_party(), *recv_buffer);
        const auto& round_ot_mask = recv_buffer;
        auto tmp = tensor_factory()->template create<int64_t>(shape);
        val_t->bitwise_xor(val_f, tmp.get());
        send_ot(tmp.get(), round_ot_mask.get(), send_buffer.get(), delta.get());
        net()->send(next_party(), *send_buffer);

        send_ot_mask(choice, send_buffer1.get(), key.get());

        net()->send(next_party(), *send_buffer1);
        net()->recv(next_party(), *recv_buffer1);

        const auto& round_ot_msg = recv_buffer1;
        recv_ot(choice, key.get(), round_ot_msg.get(), msg.get());
    } else {

        send_ot_mask(choice, send_buffer1.get(), key.get());

        net()->send(next_party(), *send_buffer1);
        net()->recv(next_party(), *recv_buffer1);

        const auto& round_ot_msg = recv_buffer1;
        recv_ot(choice, key.get(), round_ot_msg.get(), msg.get());

        net()->recv(next_party(), *recv_buffer);

        const auto& round_ot_mask = recv_buffer;
        auto tmp = tensor_factory()->template create<int64_t>(shape);
        val_t->bitwise_xor(val_f, tmp.get());
        send_ot(tmp.get(), round_ot_mask.get(), send_buffer.get(), delta.get());

        net()->send(next_party(), *send_buffer);
    }

    // val_f[i] ^ delta[i] ^ msg[i] ^ choice[i] * (val_t[i] ^ val_f[i])

    val_t->bitwise_xor(val_f, ret_.get());

    std::transform(choice->data(), choice->data() + choice->numel(),
                    ret_->data(), ret_->data(), [](u8 a, int64_t b) {
                        return (a * b);
                    });
    auto tmp = tensor_factory()->template create<int64_t>(shape);
    val_f->bitwise_xor(delta.get(), tmp.get());
    tmp->bitwise_xor(msg.get(), tmp.get());
    tmp->bitwise_xor(ret_.get(), ret);
}

inline void if_then_else_bc(TensorBlock* cond,
                                const TensorBlock* t_int,
                                const TensorBlock* f_int,
                                TensorAdapter<int64_t>* ret) {
    PADDLE_ENFORCE_EQ(cond->numel() / cond->shape()[0],
                      ret->numel(),
                      "input of condition's numel no match with return.");
    PADDLE_ENFORCE_EQ(t_int->numel() / t_int->shape()[0] / t_int->shape()[1],
                      ret->numel(),
                      "input of true val's numel no match with return.");
    PADDLE_ENFORCE_EQ(f_int->numel() / f_int->shape()[0] / f_int->shape()[1],
                      ret->numel(),
                      "input of false val's numel no match with return.");

    // convert gc input to bc 
    auto lsb_cond = tensor_factory()->create<u8>(ret->shape());
    auto lsb_t_int = tensor_factory()->template create<int64_t>(ret->shape());
    auto lsb_f_int = tensor_factory()->template create<int64_t>(ret->shape());
    block_lsb(cond, lsb_cond.get());
    lsb(t_int, lsb_t_int.get());
    lsb(f_int, lsb_f_int.get());

    // cal bc_mux
    bc_mux(lsb_cond.get(), lsb_t_int.get(), lsb_f_int.get(), ret);
}
/*
template<size_t N>
void gc_mul(const TensorBlock* rhs, const TensorBlock* rhs,
                 TensorBlock* ret) const {
    PADDLE_ENFORCE_EQ(lhs->numel(), ret->numel(),
                        "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->numel(), ret->numel(),
                        "input of rhs's numel no match with return.");

    std::vector<size_t> shape = lhs->shape();

    const unsigned int full_size = size + N;
    std::vector<size_t> shape_mul = shape;
    shape_mul[0] = full_size;
    //TensorBlock<N> l_(shape_mul);
    //TensorBlock<N> r_(shape_mul);
    //TensorBlock<N> res_(shape_mul);
    auto l_ = create_gc_share(shape_mul);
    auto r_ = create_gc_share(shape_mul);
    auto res_ = create_gc_share(shape_mul);

    for (int i = 0; i < size; i += 1) {
        (*lhs)[i]->copy((*l_)[i].get());
        (*rhs)[i]->copy((*r_)[i].get());
    }

    for (int i = 0; (unsigned)i < N; i += 1) {
        (*lhs)[size - 1]->copy((*l_)[size + i]);
        (*rhs)[size - 1]->copy((*r_)[size + i]);
    }

    mul_full(res_.get(), l_.get(), r_.get(), full_size);

    auto ret_ = tensor_factory()->template create<int64_t>(shape);
    res_->slice(N, full_size, ret_.get());
    ret_->copy(ret);s
}
*/

template<typename T, size_t N>
void FixedPointTensor<T, N>::gc_div(const TensorBlock* lhs, const TensorBlock* rhs, TensorBlock* ret) {

    PADDLE_ENFORCE_EQ(lhs->numel(), ret->numel(),
                    "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(rhs->numel(), ret->numel(),
                    "input of rhs's numel no match with return.");
    
    auto shape = lhs->shape();
    size_t size = shape[0];
    auto i1 = create_gc_share(shape);
    auto i2 = create_gc_share(shape);

    abs(lhs, i1.get());
    abs(rhs, i2.get());

    auto bit_shape = shape;
    bit_shape.erase(bit_shape.begin());

    auto sign = create_gc_share(bit_shape);
    (*lhs)[size - 1]->bitwise_xor((*rhs)[size - 1].get(), sign.get());

    const size_t full_size = shape[0] + N;

    auto full_shape = shape;
    full_shape[0] = full_size;

    auto l_ = create_gc_share(full_shape);
    auto r_ = create_gc_share(full_shape);
    auto res_ = create_gc_share(full_shape);

    std::copy(i1->data(), i1->data() + i1->numel(),
                l_->data() + N * l_->numel() / full_size);
    std::copy(i2->data(), i2->data() + i2->numel(),
                r_->data());

    div_full(res_.get(), nullptr, l_.get(), r_.get());

    auto q_sign = create_gc_share(bit_shape);
    (*res_)[size - 1]->copy(q_sign.get());

    auto nan = create_gc_share(shape);
    for (int i = 0; i < size - 1; ++i) {
        q_sign->copy((*nan)[i].get());
    }

    garbled_not(q_sign.get(), (*nan)[size - 1].get());

    privc::if_then_else(res_.get(), nan.get(), res_.get(), size, q_sign.get());

    cond_neg(sign.get(), res_.get(), res_.get());

    auto tmp = create_gc_share(bit_shape);

    garbled_and(sign.get(), q_sign.get(), tmp.get());
    (*res_)[0]->bitwise_xor(tmp.get(), (*res_)[0].get());

    std::copy(res_->data(), res_->data() + ret->numel(),
                ret->data());
}

inline void relu(const TensorBlock* lhs, TensorBlock* ret, size_t N) {
    PADDLE_ENFORCE_EQ(lhs->numel(), ret->numel(),
                        "input numel no match.");

    auto shape = lhs->shape();

    auto zero = tensor_factory()->template create<int64_t>(shape);
    privc::to_gc_num(0.0, zero.get(), N);
    auto bit_shape = shape;
    bit_shape.erase(bit_shape.begin());

    auto cmp = create_gc_share(bit_shape);

    geq(zero.get(), lhs, cmp.get());
    if_then_else(cmp.get(), zero.get(), lhs, ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::relu_bc(const TensorBlock* lhs, TensorAdapter<int64_t>* ret) {
    auto shape = lhs->shape();
    PADDLE_ENFORCE_EQ(lhs->numel() / shape[0] / shape[1],
                ret->numel(), "input numel no match.");
    auto zero = tensor_factory()->template create<int64_t>(shape);

    privc::to_gc_num(0.0, zero.get(), N);
    auto bit_shape = shape;
    bit_shape.erase(bit_shape.begin());

    auto cmp = create_gc_share(bit_shape);

    geq(zero.get(), lhs, cmp.get());
    if_then_else_bc(cmp.get(), zero.get(), lhs, ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::logistic(const TensorBlock* lhs, TensorBlock* ret) {
    auto shape = lhs->shape();
    PADDLE_ENFORCE_EQ(lhs->numel(),
                ret->numel(), "input numel no match.");
    auto bit_shape = shape;
    bit_shape.erase(bit_shape.begin());

    auto one = create_gc_share(shape);
    auto half = create_gc_share(shape);
    privc::to_gc_num(1.0, one.get(), N);
    privc::to_gc_num(0.5, half.get(), N);

    auto tmp = create_gc_share(shape);
    gc_add(lhs, half.get(), tmp.get());

    privc::relu(tmp.get(), tmp.get(), N);
    auto cmp = create_gc_share(bit_shape);
    geq(one.get(), tmp.get(), cmp.get());
    if_then_else(cmp.get(), tmp.get(), one.get(), ret);
}

inline void get_row_element(int row, const TensorBlock* share, TensorBlock* ret) {
    auto shape = share->shape();
    auto gc_element_size = sizeof(int64_t) * _g_block_size_expand * 8;
    
    auto num_row = shape[2];
    auto num_col = shape[3];
    PADDLE_ENFORCE_GT(num_row, row, "input row large than total row.");

    std::copy(share->data() + row * num_col * gc_element_size,
              share->data() + (row + 1) * num_col * gc_element_size,
              ret->data());
}

inline void get_element_from_vector(int col,
                TensorBlock* share_v, TensorBlock* ret) {
    auto shape = share_v->shape();
    auto gc_element_size = sizeof(int64_t) * _g_block_size_expand * 8;
    
    auto num_col = shape[0];
    PADDLE_ENFORCE_GT(num_col, col, "input col large than total col.");
    PADDLE_ENFORCE_EQ(ret->numel(),
            gc_element_size, "input numel no match.");
    std::copy(share_v->data() + col * gc_element_size,
              share_v->data() + (col + 1) * gc_element_size,
              ret->data());
}

inline block* get_bit_element(int row, int col,
                TensorBlock* bit_tensor) {
    auto shape = bit_tensor->shape();
    
    auto num_col = shape[2];
    PADDLE_ENFORCE_GT(num_col, col, "input col large than total col.");

    block* ret_ptr = reinterpret_cast<block*>(bit_tensor->data()) + row * num_col + col;
    return ret_ptr;
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::argmax_one_hot(const TensorBlock* op,
                TensorBlock* ret) {
    PADDLE_ENFORCE_EQ(ret->shape()[0], 1, "1 size (bit) is enough for ret");
    auto gc_shape = op->shape();
    auto val_shape = gc_shape;
    val_shape.erase(val_shape.begin());
    val_shape.erase(val_shape.begin());
    auto one_bit_shape = get_block_shape({1});

    auto shape_row{val_shape[1]};

    //TensorBlock max_one_hot(get_block_shape(val_shape));
    auto max_one_hot = create_gc_share(get_block_shape(val_shape));

    auto true_ = tensor_factory()->template create<u8>({1});
    *(true_->data()) = 1;

    // enumerate matrix column
    for (int i = 0; i < val_shape[0]; ++i) {
        // assign first cmp result to true
        block* cmp_i_0 = get_bit_element(i, 0, max_one_hot.get());
        auto bit_true = create_gc_share(one_bit_shape);
        to_gc_bit(true_.get(), 0, bit_true.get());
        *cmp_i_0 = *reinterpret_cast<const block*>(bit_true->data());

        // enumerate each element in each column
        auto row_i = tensor_factory()->template create<int64_t>(shape_row);
        get_row_element(i, op, row_i.get());

        auto max = create_gc_share(get_gc_shape({1}));
        get_element_from_vector(0, row_i.get(), max.get());
        for(int j = 1; j < val_shape[1]; ++j) {
            auto op_i_j = create_gc_share(get_gc_shape({1}));
            get_element_from_vector(j, row_i.get(), op_i_j.get());

            auto cmp = create_gc_share(one_bit_shape);
            geq(op_i_j.get(), max.get(), cmp.get());
            block* cmp_i_j = get_bit_element(i, j, max_one_hot.get());
            *cmp_i_j = *reinterpret_cast<const block*>(cmp->data());
            if_then_else(cmp.get(), op_i_j.get(), max.get(), max.get());
        }
    }

    // find max index
    auto false_ = tensor_factory()->template create<u8>({1});
    *(false_->data()) = 0;

    for (int i = 0; i < val_shape[0]; ++i) {
        auto has_found = create_gc_share(one_bit_shape);
        to_gc_bit(false_.get(), 0, has_found.get());
        for (int j = val_shape[1] - 1; j >= 0; --j) {
            // found = has_found || bit[j]
            block* bit_share_i_j = get_bit_element(i, j, max_one_hot.get());
            auto bit_i_j = create_gc_share(one_bit_shape);

            *reinterpret_cast<block*>(bit_i_j->data()) = *bit_share_i_j;

            auto found = create_gc_share(one_bit_shape);
            has_found->bitwise_xor(bit_i_j.get(), found.get());
            // bit[j] = !has_found & bit[j]
            auto tmp = create_gc_share(one_bit_shape);
            garbled_not(has_found.get(), tmp.get());
            garbled_and(tmp.get(), bit_i_j.get(), bit_i_j.get());
            *bit_share_i_j = *reinterpret_cast<const block*>(bit_i_j->data());
            // has_found = found
            *reinterpret_cast<block*>(has_found->data()) =
                        *reinterpret_cast<const block*>(found->data());
        }
    }
    max_one_hot->copy(ret);
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::to_ac_num(const TensorAdapter<int64_t>* input,
               TensorAdapter<int64_t>* ret) {
    // convert boolean share to arithmetic share
    PADDLE_ENFORCE_EQ(input->numel(), ret->numel(), "input numel no match.");
    const size_t word_width = sizeof(int64_t) * 8; // 8 bit for 1 byte

    auto shape = input->shape();

    auto gc_shape = get_gc_shape(shape, word_width);
    auto block_shape = get_block_shape(shape);
    auto buffer_shape = gc_shape;
    buffer_shape.erase(buffer_shape.begin() + 1);
    auto ret_ = tensor_factory()->template create<int64_t>(shape);
    std::for_each(ret_->data(), ret_->data() + ret_->numel(),
                  [](int64_t& a) { a = 0; });

    if (party() == 0) {
        auto s1_buffer = tensor_factory()->template create<int64_t>(buffer_shape);
        auto ot_mask = tensor_factory()->template create<int64_t>(gc_shape);
        net()->recv(next_party(), *ot_mask);
        for (size_t idx = 0; idx < word_width; ++idx) {
            // sender
            // with ot input (s0, s1)
            // s0 = (1 - x[i])* 2^i - r_i, s1 = x[i]*2^i - r_i
            auto round_ot_mask = tensor_factory()->template create<int64_t>(block_shape);
            ot_mask->slice(idx, idx + 1, round_ot_mask.get());
            round_ot_mask->reshape(block_shape);

            auto q = tensor_factory()->template create<int64_t>(block_shape);
            ot()->ot_sender().get_ot_instance(q.get());

            auto base_ot_choice = tensor_factory()->template create<int64_t>(block_shape);
            auto tmp = tensor_factory()->template create<int64_t>(block_shape);
            ot()->base_ot_choice(base_ot_choice.get());
            round_ot_mask->bitwise_and(base_ot_choice.get(), tmp.get());
            q->bitwise_xor(tmp.get(), q.get());

            auto s_first = tensor_factory()->template create<int64_t>(block_shape);
            auto s_second = tensor_factory()->template create<int64_t>(block_shape);
            q->bitwise_xor(base_ot_choice.get(), tmp.get());
            std::pair<TensorBlock*, TensorBlock*> x_pair({q.get(), tmp.get()});
            std::pair<TensorBlock*, TensorBlock*> s_pair({s_first.get(), s_second.get()});
            common::hash_blocks(x_pair, s_pair);

            auto s0 = tensor_factory()->template create<int64_t>(shape);
            block_to_int64(s_first.get(), s0.get());
            
            auto bit_mask = tensor_factory()->template create<int64_t>(shape);
            std::for_each(bit_mask->data(), bit_mask->data() + bit_mask->numel(),
                            [&idx](int64_t& a) { a = ((int64_t)1 << idx); });
            auto ai = tensor_factory()->template create<int64_t>(shape);
            input->bitwise_and(bit_mask.get(), ai.get());
            auto r = tensor_factory()->template create<int64_t>(shape);
            ai->sub(s0.get(), r.get());

            auto s1 = tensor_factory()->template create<int64_t>(shape);
            block_to_int64(s_second.get(), s1.get());

            ai->bitwise_xor(bit_mask.get(), ai.get());
            ai->sub(r.get(), ai.get());
            s1->bitwise_xor(ai.get(), s1.get());

            auto s1_buffer_s = tensor_factory()->template create<int64_t>(shape);
            s1_buffer->slice(idx, idx + 1, s1_buffer_s.get());

            s1->copy(s1_buffer_s.get());
            // ret_ =  sum_i{r_i}
            ret_->add(r.get(), ret_.get());
        }
        net()->send(next_party(), *s1_buffer);

    } else {
        // as ot recver
        // with choice bit x[i]
        auto ot_masks = tensor_factory()->template create<int64_t>(gc_shape);
        auto t0_buffer = tensor_factory()->template create<int64_t>(gc_shape);
        auto ot_msg = tensor_factory()->template create<int64_t>(buffer_shape);
        auto& ot_ext_recver = ot()->ot_receiver();
        gen_ot_masks(ot_ext_recver, input, ot_masks.get(), t0_buffer.get());

        net()->send(next_party(), *ot_masks);

        net()->recv(next_party(), *ot_msg);

        for (u64 idx = 0; idx < word_width; idx += 1) {
            auto round_ot_msg = tensor_factory()->template create<int64_t>(shape);
            ot_msg->slice(idx, idx + 1, round_ot_msg.get());
            round_ot_msg->reshape(shape);

            auto t0_buffer_s = tensor_factory()->template create<int64_t>(block_shape);
            t0_buffer->slice(idx, idx + 1, t0_buffer_s.get());

            auto t0_hash = tensor_factory()->template create<int64_t>(block_shape);
            common::hash_block(t0_buffer_s.get(), t0_hash.get());

            auto key = tensor_factory()->template create<int64_t>(shape);
            block_to_int64(t0_hash.get(), key.get());

            auto tmp = tensor_factory()->template create<int64_t>(shape);
            round_ot_msg->bitwise_xor(key.get(), tmp.get());
            auto cond = tensor_factory()->create<int64_t>(shape);
            std::transform(input->data(), input->data() + input->numel(),
                            cond->data(), [&idx] (int64_t a) {
                                return ((a >> idx) & 1);
                            });
            if_then_else_plain(false, cond.get(), tmp.get(), key.get(), tmp.get());
            // ret_ = sum_i{x[i] * 2^i} - sum_i{r_i}
            ret_->add(tmp.get(), ret_.get());
        }
    }
    ret_->copy(ret);
}


template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::relu_impl(FixedPointTensor<T, N>* ret,
                                       const Type2Type<int64_t>) const {
    PADDLE_ENFORCE_EQ(ret->numel(), numel(), "input numel mot match.");
    // ac to gc
    auto gc_shape = get_gc_shape(shape());
    auto x = tensor_factory()->template create<int64_t>(gc_shape);
    auto y = tensor_factory()->template create<int64_t>(gc_shape);
    auto gc = tensor_factory()->template create<int64_t>(gc_shape);

    to_gc_num(share(), 0, x.get());
    to_gc_num(share(), 1, y.get());


    gc_add(x.get(), y.get(), gc.get());
    // relu bc
    auto ret_bc = tensor_factory()->template create<int64_t>(shape());
    relu_bc(gc.get(), ret_bc.get());
    // bc to ac
    to_ac_num(ret_bc.get(), ret->mutable_share());

}

template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::sigmoid_impl(FixedPointTensor<T, N>* ret,
                                       const Type2Type<int64_t>) const {
    PADDLE_ENFORCE_EQ(ret->numel(), numel(), "input numel mot match.");
    // ac to gc
    auto gc_shape = get_gc_shape(shape());
    auto x = tensor_factory()->template create<int64_t>(gc_shape);
    auto y = tensor_factory()->template create<int64_t>(gc_shape);
    auto gc = tensor_factory()->template create<int64_t>(gc_shape);

    to_gc_num(share(), 0, x.get());
    to_gc_num(share(), 1, y.get());

    gc_add(x.get(), y.get(), gc.get());
    // gc logistic
    auto ret_gc = tensor_factory()->template create<int64_t>(gc_shape);
    logistic(gc.get(), ret_gc.get());
    // gc to ac
    auto bc_shape = gc_shape;
    bc_shape.erase(bc_shape.begin());
    bc_shape.erase(bc_shape.begin());
    auto res_lsb = tensor_factory()->template create<T>(bc_shape);
    lsb(ret_gc.get(), res_lsb.get());
    to_ac_num(res_lsb.get(), ret->mutable_share());
}

template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::argmax_impl(FixedPointTensor<T, N>* ret,
                                       const Type2Type<int64_t>) const {
    PADDLE_ENFORCE_EQ(ret->shape()[1], shape()[1],
                      "lhs column not match with return column.");
    PADDLE_ENFORCE_EQ(ret->numel(), numel(),
                      "input numel mot match with return.");
    // ac to gc
    auto gc_shape = get_gc_shape(shape());
    auto x = tensor_factory()->template create<int64_t>(gc_shape);
    auto y = tensor_factory()->template create<int64_t>(gc_shape);
    auto gc = tensor_factory()->template create<int64_t>(gc_shape);

    to_gc_num(share(), 0, x.get());
    to_gc_num(share(), 1, y.get());

    gc_add(x.get(), y.get(), gc.get());
    // gc argmax
    auto ret_gc_shape = get_block_shape(shape());
    // 1 bit is enough for argmax ret
    ret_gc_shape.insert(ret_gc_shape.begin(), 1);
    auto ret_gc = tensor_factory()->template create<int64_t>(ret_gc_shape);
    argmax_one_hot(gc.get(), ret_gc.get());
    // gc to ac
    auto ret_ = tensor_factory()->template create<int64_t>(ret->shape());

    lsb(ret_gc.get(), ret_.get());
    to_ac_num(ret_.get(), ret_.get());
    // to fixedpoint number
    std::transform(ret_->data(), ret_->data() + ret_->numel(),
                   ret->mutable_share()->data(),
                   [] (T a) {
                       // int to fixedpoint
                       return (a << N); 
                    });
}

template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::long_div_impl(const FixedPointTensor<T, N>* rhs,
                                       FixedPointTensor<T, N>* ret,
                                       const Type2Type<int64_t>) const {
    PADDLE_ENFORCE_EQ(ret->numel(), numel(),
            "input of lhs's numel no match with return.");
    PADDLE_ENFORCE_EQ(ret->numel(), rhs->numel(),
            "input of rhs's numel no match with return.");
    // ac to gc
    auto gc_shape = get_gc_shape(shape());
    auto l_x = tensor_factory()->template create<int64_t>(gc_shape);
    auto l_y = tensor_factory()->template create<int64_t>(gc_shape);
    auto l_gc = tensor_factory()->template create<int64_t>(gc_shape);

    to_gc_num(share(), 0, l_x.get());
    to_gc_num(share(), 1, l_y.get());

    gc_add(l_x.get(), l_y.get(), l_gc.get());


    auto r_x = tensor_factory()->template create<int64_t>(gc_shape);
    auto r_y = tensor_factory()->template create<int64_t>(gc_shape);
    auto r_gc = tensor_factory()->template create<int64_t>(gc_shape);

    to_gc_num(rhs->share(), 0, r_x.get());
    to_gc_num(rhs->share(), 1, r_y.get());
 
    gc_add(r_x.get(), r_y.get(), r_gc.get());

    // gc logistic
    auto ret_gc = tensor_factory()->template create<int64_t>(gc_shape);
    gc_div(l_gc.get(), r_gc.get(), ret_gc.get());
    // gc to ac
    auto bc_shape = gc_shape;
    bc_shape.erase(bc_shape.begin());
    bc_shape.erase(bc_shape.begin());
    auto res_lsb = tensor_factory()->template create<T>(bc_shape);

    lsb(ret_gc.get(), res_lsb.get());
    to_ac_num(res_lsb.get(), ret->mutable_share());
}

// reduce last dim
template <typename T, size_t N>
void FixedPointTensor<T, N>::reduce(FixedPointTensor<T, N>* ret) const {
    std::vector<size_t> shape_reduce = ret->shape();
    auto shape = this->shape();
    size_t numel = ret->numel();
    for (int i = 0; i < this->numel(); ++i) {
        T* ret_ptr = ret->mutable_share()->data() + i;
        *ret_ptr = 0;
        std::for_each(share()->data() + i * shape[shape.size() - 1],
                      share()->data() + (i + 1) * shape[shape.size() - 1],
                      [ret_ptr](T a) { *ret_ptr += a; });
    }
}

template<typename T, size_t N>
template<typename T_>
void FixedPointTensor<T, N>::softmax_impl(FixedPointTensor<T, N>* ret,
                                     bool use_relu,
                                     const Type2Type<int64_t>) const {
    auto tmp = tensor_factory()->template create<int64_t>(shape());
    FixedPointTensor<T, N> x(tmp.get());
    if (use_relu) {
        this->relu(&x);
    } else {
        this->exp(&x);
    }

    auto tmp1 = tensor_factory()->template create<int64_t>({shape()[1]});
    FixedPointTensor<T, N> sum(tmp1.get());
    x.reduce(&sum);

    auto sum_ext = tensor_factory()->template create<int64_t>(shape());
    const T* sum_ptr = sum.share()->data();
    for (int i = 0; i < shape()[0]; ++i) {
        T* col_ptr = sum_ext->data() + i * shape()[1];
        //std::transform(sum_ptr, sum_ptr + shape()[0], col_ptr,
        //               [](T a) { return a; });
        std::for_each(col_ptr, col_ptr + shape()[1],
                       [sum_ptr, i]( T& a) { a = *(sum_ptr + i); });
    }
    FixedPointTensor<T, N> sum_ext_fixed(sum_ext.get());
    x.long_div(&sum_ext_fixed, ret);
}

} // namespace privc
