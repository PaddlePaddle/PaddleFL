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

#include "paddle/fluid/framework/op_registry.h"

#include "mpc_batch_norm_op.h"

#include "core/common/paddle_tensor_impl.cu.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void cu_expand(T* dst, const T* src, int S, int N, int C, int sample_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    while (col < S * N * C * sample_size) {

        int share = col / (N * C * sample_size);

        int nc = (col / sample_size) % (N * C);

        dst[col] = src[nc % C + share * C];

        col += blockDim.x * gridDim.x;
    }
}

template <typename T>
struct Expand<platform::CUDADeviceContext, T> {

    void operator()(const Tensor* input, Tensor* output, int S, int N, int C, int sample_size) {
        // Expand tensor into specified shape
        // input shape: {S, C}
        // outout shape: {S, N, C, H, W}, sample_size = H * W
        const T* input_data = input->data<T>();
        T* output_data = output->data<T>();

        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((S * N * C * sample_size + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);

        cu_expand<T><<<grid_size, block_size, 0, mpc::AbstractContext::_s_stream>>>(
            output_data, input_data, S, N, C, sample_size);
    }

};

template <typename T>
__global__ void cu_compute_sum(T* dst, const T* src, int S, int N, int C, int sample_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    while (col < S * C) {
        int s = col / C;
        int c = col % C;

        dst[col] = 0;

        for (int i = 0; i < N * sample_size; ++i) {
            int n = i / sample_size;
            int i_ = i % sample_size;
            dst[col] += src[s * N * C * sample_size
                + n * C * sample_size
                + c * sample_size
                + i_];
        }

        col += blockDim.x * gridDim.x;
    }
}

template <typename T>
struct ComputeSum<platform::CUDADeviceContext, T> {
    void operator()(const Tensor* input, int C, Tensor* sum, const framework::ExecutionContext &ctx) {
        // Compute sum of each channel
        // input shape: {S, N, C, H, W}
        // output shape: {S, C}
        // H and W is optional, compute the sum of each channel.
        int S = input->dims()[0];
        int N = input->dims()[1];
        int sample_size = input->numel() / (S * N * C);

        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((S * C + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);

        cu_compute_sum<T><<<grid_size, block_size, 0, mpc::AbstractContext::_s_stream>>>(
            sum->data<T>(), input->data<T>(), S, N, C, sample_size);
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    mpc_batch_norm, ops::MpcBatchNormKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    mpc_batch_norm_grad, ops::MpcBatchNormGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
