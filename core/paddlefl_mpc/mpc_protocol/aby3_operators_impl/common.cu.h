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

#pragma once

#include "paddle/fluid/framework/tensor.h"
#include "core/paddlefl_mpc/mpc_protocol/mpc_operators.h"
#include "core/common/paddle_tensor_impl.cu.h"
#include "core/privc3/fixedpoint_tensor.h"
#include "core/privc3/boolean_tensor.h"

namespace paddle {
namespace operators {
namespace aby3 {

using Tensor = paddle::framework::Tensor;
// TODO: decide scaling factor
const size_t ABY3_SCALING_FACTOR = paddle::mpc::FIXED_POINTER_SCALING_FACTOR;
static const size_t SHARE_NUM = 2;

using FixedTensor = ::aby3::FixedPointTensor<int64_t, ABY3_SCALING_FACTOR>;
using BoolTensor = ::aby3::BooleanTensor<int64_t>;
using CudaPaddleTensor = common::CudaPaddleTensor<int64_t>;

template <typename T>
std::tuple<
    std::shared_ptr<T>,
    std::shared_ptr<CudaPaddleTensor>,
    std::shared_ptr<CudaPaddleTensor> > from_tensor(const Tensor* t);

std::tuple<
    std::shared_ptr<FixedTensor>,
    std::shared_ptr<CudaPaddleTensor>,
    std::shared_ptr<CudaPaddleTensor> > from_tensor(const Tensor* t);

struct GetMidDims {
    inline HOST void operator()(const framework::DDim &x_dims,
                         const framework::DDim &y_dims, const int axis,
                         int *pre, int *n, int *post)  {
        *pre = 1;
        *n = 1;
        *post = 1;
        for (int i = 1; i < axis + 1; ++i) {
            (*pre) *= x_dims[i];
        }

        for (int i = 1; i < y_dims.size(); ++i) {
            PADDLE_ENFORCE_EQ(x_dims[i + axis], y_dims[i],
                              "Broadcast dimension mismatch.");
            (*n) *= y_dims[i];
        }

        for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
            (*post) *= x_dims[i];
        }
    }
};

} // aby3
} // operators
} // paddle

//#endif
