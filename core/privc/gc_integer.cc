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

#include "gc_integer.h"

#include <algorithm>
#include <vector>

#include "gc_bit.h"
#include "common_utils.h"
#include "core/common/paddle_tensor.h"

namespace privc {

void add_full(IntegerTensor *dest, BitTensor *carry_out,
              const IntegerTensor *op1, const IntegerTensor *op2,
              const BitTensor *carry_in, int size, size_t pos_dest = 0,
              size_t pos_op1 = 0, size_t pos_op2 = 0) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> tmp;
    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(tensor_factory()->template create<int64_t>(bit_shape));
    }
    BitTensor carry(tmp[0]);
    BitTensor bxc(tmp[1]);
    BitTensor axc(tmp[2]);
    BitTensor t(tmp[3]);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (carry_in && carry_out) {
            carry_in->share()->copy(carry_out->mutable_share());
        }
        return;
    }
    if (carry_in) {
        carry_in->share()->copy(carry.mutable_share());
    } else {
        carry.set_false();
    }
    // skip AND on last bit if carry_out==NULL
    skip_last = (carry_out == nullptr);
    while (size-- > skip_last) {
        (*op1)[i + pos_op1]->bitwise_xor(&carry, &axc);
        (*op2)[i + pos_op2]->bitwise_xor(&carry, &bxc);
        (*op1)[i + pos_op1]->bitwise_xor(&bxc, (*dest)[i + pos_dest].get());
        axc.bitwise_and(&bxc, &t);
        carry.bitwise_xor(&t, &carry);
        ++i;
    }
    if (carry_out != nullptr) {
        carry.share()->copy(carry_out->mutable_share());
    } else {
        carry.bitwise_xor((*op2)[i + pos_op2].get(), (*dest)[i + pos_dest].get());
        (*dest)[i + pos_dest]->
            bitwise_xor((*op1)[i + pos_op1].get(), (*dest)[i + pos_dest].get());
    }
        
}

void sub_full(IntegerTensor *dest, BitTensor *borrow_out,
              const IntegerTensor *op1, const IntegerTensor *op2,
              const BitTensor *borrow_in, int size) {
    auto bit_shape = dest->shape();
    bit_shape.erase(bit_shape.begin());
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> tmp;
    for (int i = 0; i < 4; ++i) {
        tmp.emplace_back(tensor_factory()->template create<int64_t>(bit_shape));
    }
    BitTensor borrow(tmp[0]);
    BitTensor bxc(tmp[1]);
    BitTensor bxa(tmp[2]);
    BitTensor t(tmp[3]);

    int skip_last = 0;
    int i = 0;

    if (size == 0) {
        if (borrow_in && borrow_out) {
            *borrow_out = *borrow_in;
        }
        return;
    }
    if (borrow_in) {
        borrow = *borrow_in;
    } else {
        borrow.set_false();
    }
    // skip AND on last bit if borrow_out==NULL
    skip_last = (borrow_out == nullptr);
    while (size-- > skip_last) {
        (*op1)[i]->bitwise_xor((*op2)[i].get(), &bxa);
        borrow.bitwise_xor((*op2)[i].get(), &bxc);
        bxa.bitwise_xor(&borrow, (*dest)[i].get());
        bxa.bitwise_and(&bxc, &t);
        borrow.bitwise_xor(&t, &borrow);
        ++i;
    }
    if (borrow_out != nullptr) {
        *borrow_out = borrow;
    } else {
        (*op1)[i]->bitwise_xor((*op2)[i].get(), (*dest)[i].get());
        (*dest)[i]->bitwise_xor(&borrow, (*dest)[i].get());
    }
}

void mul_full(IntegerTensor *dest, const IntegerTensor *op1,
              const IntegerTensor *op2, int size) {
    IntegerTensor sum(dest->shape());
    IntegerTensor tmp(dest->shape());

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size - i; ++k) {
            (*op1)[k]->bitwise_and((*op2)[i].get(), tmp[k].get());
        }
        add_full(&sum, nullptr, &sum, &tmp, nullptr, size - i, i, i, 0);
    }
    // calc dest sign
    (*op1)[size - 1]->bitwise_xor((*op2)[size - 1].get(), sum[size - 1].get());
    sum.share()->copy(dest->mutable_share());
}

void if_then_else(IntegerTensor* dest, const IntegerTensor *tsrc,
                  const IntegerTensor *fsrc, int size,
                  BitTensor* cond) {
    int i = 0;
    while (size-- > 0) {
        BitTensor x((*tsrc)[i]->shape());
        (*tsrc)[i]->bitwise_xor((*fsrc)[i].get(), &x);
        BitTensor a((*tsrc)[i]->shape());
        cond->bitwise_and(&x, &a);
        a.bitwise_xor((*fsrc)[i].get(), (*dest)[i].get());
        ++i;
    }
}

IntegerTensor::IntegerTensor(const TensorAdapter<int64_t>* input, size_t party_in) {
    _length = sizeof(int64_t) * 8;
    auto shape = input->shape();
    auto bit_shape = shape;
    bit_shape.insert(bit_shape.begin(), _length);

    auto gc_shape = bit_shape;
    gc_shape.insert(gc_shape.begin() + 1, _g_block_size_expand);

    auto input_bits = tensor_factory()->template create<u8>(bit_shape);
    to_bits(input, input_bits.get());

    _bits_tensor = tensor_factory()->template create<int64_t>(gc_shape);
    if (party_in == 0) {
        if (party() == 0) {
            auto to_send = tensor_factory()->template create<int64_t>(gc_shape);
            privc_ctx()->template gen_random_private(*to_send);
            to_send->copy(_bits_tensor.get());

            auto mask_val = tensor_factory()->template create<int64_t>(gc_shape);
            auto garbled_delta = tensor_factory()->template create<int64_t>(gc_shape);
            auto zero_block_tensor = tensor_factory()->template create<int64_t>(gc_shape);
            std::for_each(zero_block_tensor->data(),
                          zero_block_tensor->data() + zero_block_tensor->numel(),
                          [](int64_t& a) { a = 0; });
            ot()->garbled_delta(garbled_delta.get());
            if_then_else_plain(input_bits.get(), garbled_delta.get(), zero_block_tensor.get(), mask_val.get());

            to_send->bitwise_xor(mask_val.get(), to_send.get());
            net()->send(next_party(), *to_send);

        } else {
            //TODO: recv without shape
            net()->recv(next_party(), *_bits_tensor);
        }
    } else {
        garbled_share(input_bits.get(), _bits_tensor.get());
    }
}

std::shared_ptr<BitTensor> IntegerTensor::operator[](int index) {
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    auto ret = tensor_factory()->template create<int64_t>(bit_shape);

    _bits_tensor->slice(index, index + 1, ret.get());
    ret->reshape(bit_shape);
    return std::make_shared<BitTensor>(ret);
}

const std::shared_ptr<BitTensor> IntegerTensor::operator[](int index) const {
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    auto ret = tensor_factory()->template create<int64_t>(bit_shape);

    _bits_tensor->slice(index, index + 1, ret.get());
    ret->reshape(bit_shape);
    return std::make_shared<BitTensor>(ret);
}


// Comparisons
void IntegerTensor::geq(const IntegerTensor *rhs, BitTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel() / shape()[0],
                      ret->share()->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel() / shape()[0],
                      ret->share()->numel(), "input numel no match.");

    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }

    IntegerTensor dest(shape());
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());
    BitTensor borrow_out(bit_shape);

    sub_full(&dest, &borrow_out, this, rhs, nullptr, size());

    (*this)[size() - 1]->bitwise_xor((*rhs)[size() - 1].get(), ret);
    ret->bitwise_xor(&borrow_out, ret);
    ret->bitwise_not(ret);
}

void IntegerTensor::equal(const IntegerTensor* rhs, BitTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel() / shape()[0], ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel() / shape()[0],
                      ret->share()->numel(), "input numel no match.");

    if (size() != rhs->size()) {
        throw std::logic_error("op len not match");
    }
    auto bit_shape = shape();
    bit_shape.erase(bit_shape.begin());

    BitTensor res(bit_shape);
    BitTensor tmp(bit_shape);

    res.bitwise_not(&res);
    for (int i = 0; i < size(); ++i) {

        (*this)[i]->bitwise_xor((*rhs)[i].get(), &tmp);
        tmp.bitwise_not(&tmp);
        res.bitwise_and(&tmp, &res);
    }
    res.share()->copy(ret->mutable_share());
}

void IntegerTensor::is_zero(BitTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel() / shape()[0],
                     ret->share()->numel(), "input numel no match.");
    for (int i = 0; i < size(); ++i) {
        ret->bitwise_or((*this)[i].get(), ret);
    }

}

void IntegerTensor::abs(IntegerTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");

    IntegerTensor res(shape());
    for (int i = 0; i < size(); ++i) {
        (*this)[size() - 1]->share()->copy(res[i]->mutable_share());
    }
    bitwise_add(&res, ret);
    ret->bitwise_xor(&res, ret);
}

void IntegerTensor::bitwise_xor(const IntegerTensor* rhs,
                                IntegerTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                      "input numel no match.");

    share()->bitwise_xor(rhs->share(), ret->mutable_share());
}

void IntegerTensor::bitwise_add(const IntegerTensor* rhs,
                                IntegerTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(size(), rhs->size(), "input size no match.");

    add_full(ret, nullptr, this, rhs, nullptr, size());
}

void IntegerTensor::bitwise_sub(const IntegerTensor* rhs,
                                IntegerTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(size(), rhs->size(), "input size no match.");

    sub_full(ret, nullptr, this, rhs, nullptr, size());
}

void IntegerTensor::bitwise_mul(const IntegerTensor* rhs,
                                IntegerTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(rhs->share()->numel(), ret->share()->numel(),
                      "input numel no match.");
    PADDLE_ENFORCE_EQ(size(), rhs->size(), "input size no match.");

    IntegerTensor ret_(shape());
    mul_full(&ret_, this, rhs, size());
    ret_.share()->copy(ret->mutable_share());
}

void IntegerTensor::bitwise_neg(IntegerTensor* ret) const {
    PADDLE_ENFORCE_EQ(share()->numel(), ret->share()->numel(),
                      "input numel no match.");

    IntegerTensor zero(shape());
    sub_full(ret, nullptr, &zero, this, nullptr, size());
}

void IntegerTensor::if_then_else(BitTensor* cond, const IntegerTensor* t_int,
                              const IntegerTensor* f_int,
                              IntegerTensor* ret) {
    PADDLE_ENFORCE_EQ(cond->share()->numel() * sizeof(int64_t) * 8,
                      ret->share()->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(t_int->share()->numel(),
                      ret->share()->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(f_int->share()->numel(),
                      ret->share()->numel(), "input numel no match.");

    IntegerTensor res(t_int->shape());
    privc::if_then_else(&res, t_int, f_int, res.size(), cond);

    res.share()->copy(ret->mutable_share());
}

void IntegerTensor::if_then_else_bc(BitTensor* cond,
                                const IntegerTensor* t_int,
                                const IntegerTensor* f_int,
                                TensorAdapter<int64_t>* ret) {
    PADDLE_ENFORCE_EQ(cond->share()->numel() / cond->shape()[0],
                      ret->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(t_int->share()->numel() / t_int->shape()[0] / t_int->shape()[1],
                      ret->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(f_int->share()->numel() / f_int->shape()[0] / f_int->shape()[1],
                      ret->numel(), "input numel no match.");

    auto lsb_cond = tensor_factory()->create<u8>(ret->shape());
    auto lsb_t_int = tensor_factory()->template create<int64_t>(ret->shape());
    auto lsb_f_int = tensor_factory()->template create<int64_t>(ret->shape());
    block_lsb(cond->share(), lsb_cond.get());
    t_int->lsb(lsb_t_int.get());
    f_int->lsb(lsb_f_int.get());
    bc_mux(lsb_cond.get(), lsb_t_int.get(), lsb_f_int.get(), ret);
}

void get_row_element(int row, const TensorBlock* share, TensorBlock* ret) {
    auto shape = share->shape();
    auto gc_element_size = sizeof(int64_t) * _g_block_size_expand * 8;
    
    auto num_row = shape[2];
    auto num_col = shape[3];
    PADDLE_ENFORCE_GT(num_row, row, "input row large than total row.");

    std::copy(share->data() + row * num_col * gc_element_size,
              share->data() + (row + 1) * num_col * gc_element_size,
              ret->data());
}

void get_element_from_vector(int col,
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

block* get_bit_element(int row, int col,
                TensorBlock* bit_tensor) {
    auto shape = bit_tensor->shape();
    
    auto num_col = shape[2];
    PADDLE_ENFORCE_GT(num_col, col, "input col large than total col.");

    block* ret_ptr = reinterpret_cast<block*>(bit_tensor->data()) + row * num_col + col;
    return ret_ptr;
}

void IntegerTensor::argmax_one_hot(
                const IntegerTensor* op,
                IntegerTensor* ret) {
    PADDLE_ENFORCE_EQ(ret->shape()[0], 1, "1 size (bit) is enough for ret");
    auto gc_shape = op->shape();
    auto val_shape = gc_shape;
    val_shape.erase(val_shape.begin());
    val_shape.erase(val_shape.begin());
    auto one_bit_shape = get_block_shape({1});

    auto shape_row{val_shape[1]};

    BitTensor max_one_hot(get_block_shape(val_shape));

    auto true_ = tensor_factory()->template create<u8>({1});
    *(true_->data()) = 1;

    // enumerate matrix column
    for (int i = 0; i < val_shape[0]; ++i) {
        // assign first cmp result to true
        block* cmp_i_0 = get_bit_element(i, 0, max_one_hot.mutable_share());
        BitTensor bit_true(true_.get(), 0);
        *cmp_i_0 = *reinterpret_cast<const block*>(bit_true.share()->data());

        // enumerate each element in each column
        auto row_i = tensor_factory()->template create<int64_t>(shape_row);
        get_row_element(i, op->share(), row_i.get());

        IntegerTensor max(get_gc_shape({1}));
        get_element_from_vector(0, row_i.get(), max.mutable_share());
        for(int j = 1; j < val_shape[1]; ++j) {
            IntegerTensor op_i_j(get_gc_shape({1}));
            get_element_from_vector(j, row_i.get(), op_i_j.mutable_share());

            BitTensor cmp(one_bit_shape);
            op_i_j.geq(&max, &cmp);
            block* cmp_i_j = get_bit_element(i, j, max_one_hot.mutable_share());
            *cmp_i_j = *reinterpret_cast<const block*>(cmp.share()->data());
            if_then_else(&cmp, &op_i_j, &max, &max);
        }
    }

    // find max index
    auto false_ = tensor_factory()->template create<u8>({1});
    *(false_->data()) = 0;

    for (int i = 0; i < val_shape[0]; ++i) {
        BitTensor has_found(false_.get(), 0);
        for (int j = val_shape[1] - 1; j >= 0; --j) {
            // found = has_found || bit[j]
            block* bit_share_i_j = get_bit_element(i, j, max_one_hot.mutable_share());
            BitTensor bit_i_j(one_bit_shape);
            *reinterpret_cast<block*>(bit_i_j.mutable_share()->data()) = *bit_share_i_j;

            BitTensor found(one_bit_shape);
            has_found.bitwise_xor(&bit_i_j, &found);
            // bit[j] = !has_found & bit[j]
            BitTensor tmp(one_bit_shape);
            has_found.bitwise_not(&tmp);
            tmp.bitwise_and(&bit_i_j, &bit_i_j);
            *bit_share_i_j = *reinterpret_cast<const block*>(bit_i_j.share()->data());
            // has_found = found
            *reinterpret_cast<block*>(has_found.mutable_share()->data()) =
                        *reinterpret_cast<const block*>(found.share()->data());
        }
    }
    // BitTensor to IntegerTensor
    max_one_hot.share()->copy(ret->mutable_share());
}

void to_ac_num(const TensorAdapter<int64_t>* input,
               TensorAdapter<int64_t>* ret) {

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

            ret_->add(r.get(), ret_.get());
        }
        net()->send(next_party(), *s1_buffer);

    } else { // as ot recver

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
            ret_->add(tmp.get(), ret_.get());
        }
    }
    ret_->copy(ret);
}

void bc_mux(const TensorAdapter<u8>* choice,
            const TensorAdapter<int64_t>* val_t,
            const TensorAdapter<int64_t>* val_f,
            TensorAdapter<int64_t>* ret) {
    PADDLE_ENFORCE_EQ(choice->numel(), ret->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(val_t->numel(), ret->numel(), "input numel no match.");
    PADDLE_ENFORCE_EQ(val_f->numel(), ret->numel(), "input numel no match.");

    auto send_ot = [](const TensorAdapter<int64_t>* diff,
                          const TensorBlock* round_ot_mask,
                          TensorAdapter<int64_t>* send_buffer,
                          TensorAdapter<int64_t>* ret) {
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

}; // namespace smc
