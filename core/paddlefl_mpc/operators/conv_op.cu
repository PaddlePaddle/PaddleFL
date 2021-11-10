/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "conv_op.h"

#include "paddle/fluid/framework/op_registry.h"

#include "core/common/paddle_tensor_impl.cu.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void cu_copy(T* dst, const T* src, size_t numel) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    while (col < numel) {
        dst[col] = src[col];
        col += blockDim.x * gridDim.x;
    }
}


template <typename T>
struct CopyData<platform::CUDADeviceContext, T> {
    void operator()(T* dst, const T* src, size_t numel) {
        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((numel + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);

        cu_copy<T><<<grid_size, block_size, 0, mpc::AbstractContext::_s_stream>>>(
            dst, src, numel);
    }
};

}// namespace paddle
}// namespace operators

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    mpc_conv2d, ops::GemmConvKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    mpc_conv2d_grad,
    ops::GemmConvGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
