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

#include "mpc_softmax_with_cross_entropy_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "core/common/paddle_tensor_impl.cu.h"
#include "core/paddlefl_mpc/mpc_protocol/abstract_context.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void cu_set_expand(T* dst, const T* src, size_t n, size_t d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    while (col < n * d) {
        int i = col / d;
        int j = col % d;
        dst[i * d + j] = src[i];
        col += blockDim.x * gridDim.x;
    }
}


template <typename T>
struct SetExpandData<paddle::platform::CUDADeviceContext, T> {
    void operator()(T* dst, const T* src, size_t n, size_t d) {
        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((n * d+ PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);
        cu_set_expand<T><<<grid_size, block_size, 0, mpc::AbstractContext::_s_stream>>>(
            dst, src, n, d);
    }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(mpc_softmax_with_cross_entropy,
                        ops::MpcSoftmaxWithCrossEntropyKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(mpc_softmax_with_cross_entropy_grad,
                       ops::MpcSoftmaxWithCrossEntropyGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
