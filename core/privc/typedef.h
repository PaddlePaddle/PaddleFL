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
//#include "crypto.h"

namespace privc {

// use alias name to distinguish TensorAdapter<block>
// and normal TensorAdapter<int64_t>
using TensorBlock = aby3::TensorAdapter<int64_t>;

template<typename T>
using TensorAdapter = aby3::TensorAdapter<T>;

typedef unsigned char u8;
typedef unsigned long long u64;

const block ZeroBlock = _mm_set_epi64x(0, 0);
const block OneBlock = _mm_set_epi64x(-1, -1);

static const size_t _g_block_size_expand = sizeof(block) / sizeof(int64_t);

} // namespace privc