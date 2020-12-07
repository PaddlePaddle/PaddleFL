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

#include "core/privc/privc_context.h"
#include "core/common/tensor_adapter_factory.h"
#include "core/common/tensor_adapter.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/common/crypto.h"
#include "core/common/naorpinkas_ot.h"
#include "core/common/ot_extension.h"

#include "paddle/fluid/platform/enforce.h"

namespace privc {

// use alias name to distinguish TensorAdapter<block>
// and normal TensorAdapter<int64_t>
using TensorBlock = common::TensorAdapter<int64_t>;

template<typename T>
using TensorAdapter = common::TensorAdapter<T>;

using u8 = common::u8;
using u64 = common::u64;

using block = common::block;

using NaorPinkasOTsender = common::NaorPinkasOTsender;
using NaorPinkasOTreceiver = common::NaorPinkasOTreceiver;

using AbstractNetwork = paddle::mpc::AbstractNetwork;
using AbstractContext = paddle::mpc::AbstractContext;

using OT = ObliviousTransfer;

template<typename T>
using OTExtSender = common::OTExtSender<T>;
template<typename T>
using OTExtReceiver = common::OTExtReceiver<T>;

static const size_t _g_block_size_expand = sizeof(block) / sizeof(int64_t);

static std::shared_ptr<AbstractContext> privc_ctx() {
  return paddle::mpc::ContextHolder::mpc_ctx();
}

static std::shared_ptr<OT> ot() {
    return std::dynamic_pointer_cast<PrivCContext>(privc_ctx())->ot();
}

static std::shared_ptr<common::TensorAdapterFactory> tensor_factory() {
    return paddle::mpc::ContextHolder::tensor_factory();
}

static std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> tripletor() {
    return std::dynamic_pointer_cast<PrivCContext>(privc_ctx())->triplet_generator();
}

static size_t party() {
    return privc_ctx()->party();
}

static size_t next_party() {
    return privc_ctx()->next_party();
}

static AbstractNetwork* net() {
  return privc_ctx()->network();
}

static std::vector<size_t> get_gc_shape(std::vector<size_t> shape,
                                              size_t size = sizeof(int64_t) * 8) {
    // using two int64_t sizes to indicate block size
    shape.insert(shape.begin(), _g_block_size_expand);
    // insert bit length
    shape.insert(shape.begin(), size);
    return shape;
}

static std::vector<size_t> get_block_shape(std::vector<size_t> shape) {
    // using two int64_t sizes to indicate block size
    shape.insert(shape.begin(), _g_block_size_expand);
    return shape;
}

static void block_to_int64(const TensorBlock* input, TensorAdapter<int64_t>* ret) {
    PADDLE_ENFORCE_EQ(input->numel(), ret->numel() * 2, "input numel no match.");
    const block* in_ptr = reinterpret_cast<const block*>(input->data());
    std::transform(in_ptr, in_ptr + ret->numel(), ret->data(),
                   [](block a) { return *(reinterpret_cast<int64_t*>(&a)); });
}

} // namespace privc