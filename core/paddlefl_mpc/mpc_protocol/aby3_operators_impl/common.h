/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Description: implementations of each virtual op according to ABY3 protocol

#pragma once

//#ifndef COMMON_H
//#define COMMON_H

#include "core/paddlefl_mpc/mpc_protocol/aby3_operators.h"

namespace paddle {
namespace operators {
namespace aby3 {

using namespace paddle::mpc;

static const size_t SHARE_NUM = 2;

template <typename T>
static std::tuple<
    std::shared_ptr<T>,
    std::shared_ptr<PaddleTensor>,
    std::shared_ptr<PaddleTensor> > from_tensor(const Tensor* t) {

        PADDLE_ENFORCE_EQ(t->dims()[0], 2);

        auto pt0 = std::make_shared<PaddleTensor>(ContextHolder::device_ctx(), t->Slice(0, 1));
        auto pt1 = std::make_shared<PaddleTensor>(ContextHolder::device_ctx(), t->Slice(1, 2));

        auto shape = pt0->shape();
        shape.erase(shape.begin());
        pt0->reshape(shape);
        pt1->reshape(shape);

        TensorAdapter<int64_t>* pt_array[2] = {pt0.get(), pt1.get()};

        auto ft = std::make_shared<T>(pt_array);

    return std::make_tuple(ft, pt0, pt1);
}

static std::tuple<
    std::shared_ptr<FixedTensor>,
    std::shared_ptr<PaddleTensor>,
    std::shared_ptr<PaddleTensor> > from_tensor(const Tensor* t) {

    return from_tensor<FixedTensor>(t);
}

} // aby3
} // operators
} // paddle

//#endif
