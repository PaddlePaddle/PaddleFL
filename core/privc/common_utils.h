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
#include "core/privc3/tensor_adapter_factory.h"
#include "../privc3/tensor_adapter.h"
#include "typedef.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"

namespace privc {

// use alias name to distinguish TensorAdapter<block>
// and normal TensorAdapter<int64_t>
//using TensorBlock = aby3::TensorAdapter<int64_t>;

//template<typename T>
//using TensorAdapter = aby3::TensorAdapter<T>;

using OT = ObliviousTransfer;

static std::shared_ptr<AbstractContext> privc_ctx();

static std::shared_ptr<OT> ot();

static std::shared_ptr<aby3::TensorAdapterFactory> tensor_factory();

static std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> tripletor();

static size_t party();

static size_t next_party();

static AbstractNetwork* net();

static std::vector<size_t> get_gc_shape(std::vector<size_t> shape,
                                              size_t size = sizeof(int64_t)) {
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
    const block* in_ptr = reinterpret_cast<const block*>(input->data());
    std::transform(in_ptr, in_ptr + ret->numel(), ret->data(),
                   [](block a) { return *(reinterpret_cast<int64_t*>(&a));});
}

} // namespace privc