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

#include "core/common/tensor_adapter_factory.h"
#include "core/common/tensor_adapter.h"
#include "core/common/crypto.h"
#include "core/common/naorpinkas_ot.h"
#include "core/common/ot_extension.h"

namespace privc {

// use alias name to distinguish TensorAdapter<block>
// and normal TensorAdapter<int64_t>
using TensorBlock = common::TensorAdapter<int64_t>;

template<typename T>
using TensorAdapter = common::TensorAdapter<T>;

using u8 = common::u8;
using u64 = common::u64;

using block = common::block;

//const block ZeroBlock = common::ZeroBlock;
//const block OneBlock = common::OneBlock;

using NaorPinkasOTsender = common::NaorPinkasOTsender;
using NaorPinkasOTreceiver = common::NaorPinkasOTreceiver;

using AbstractNetwork = paddle::mpc::AbstractNetwork;
using AbstractContext = paddle::mpc::AbstractContext;

template<typename T>
using OTExtSender = common::OTExtSender<T>;
template<typename T>
using OTExtReceiver = common::OTExtReceiver<T>;

static const size_t _g_block_size_expand = sizeof(block) / sizeof(int64_t);

} // namespace privc