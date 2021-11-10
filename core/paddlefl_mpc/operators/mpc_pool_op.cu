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

#include "mpc_pool_op.h"

#include "paddle/fluid/framework/op_registry.h"

#include "core/common/paddle_tensor_impl.cu.h"

namespace paddle {
namespace operators {

__device__ int cu_min(int op0, int op1) {
    return op0 > op1 ? op1 : op0;
}

__device__ int cu_max(int op0, int op1) {
    return op0 > op1 ? op0 : op1;
}


template <typename T, typename Func>
__global__ void cu_visit_im(
    int input_height, int out_height, int stride_height, int padding_height, int ksize_height,
    int input_width, int out_width, int stride_width, int padding_width, int ksize_width,
    const T* src, T* target, int src_stride, int target_stride, int numel, Func visitor) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    while (col < numel) {
        const T* src_ = src + col * src_stride;
        T* target_ = target + col * target_stride;

        int hstart, hend;
        int wstart, wend;

        for (size_t ph = 0; ph < out_height; ++ph) {
            hstart =  ph * stride_height - padding_height;
            hend = cu_min(hstart + ksize_height, input_height);
            hstart = cu_max(hstart, 0);

            for (size_t pw = 0; pw < out_width; ++pw) {
                wstart = pw * stride_width - padding_width;
                wend = cu_min(wstart + ksize_width, input_width);
                wstart = cu_max(wstart, 0);

                visitor(ph, pw, input_height, input_width, out_height, out_width, hstart, hend,
                        wstart, wend, src_, target_);
            }
        }
        col += blockDim.x * gridDim.x;
    }
}

template <typename T, typename Func>
struct VisitDataStrideWise<paddle::platform::CUDADeviceContext, T, Func> {
    void operator()(DDim in_dims, DDim out_dims,
                    std::vector<int>& ksize, std::vector<int>& strides, std::vector<int>& paddings,
                    const T* src, T* target, int src_stride, int target_stride, Func visitor) {
        const int share_size = in_dims[0];
        const int batch_size = in_dims[1];
        const int channel_size = in_dims[2];
        const int input_height = in_dims[3];
        const int input_width = in_dims[4];
        const int out_height = out_dims[3];
        const int out_width = out_dims[4];
        const int out_mat_numel = out_height * out_width;

        const int ksize_height = ksize[0];
        const int ksize_width = ksize[1];
        const int filter_numel = ksize_height * ksize_width;
        const int stride_height = strides[0];
        const int stride_width = strides[1];
        const int padding_height = paddings[0];
        const int padding_width = paddings[1];

        int hstart, hend;
        int wstart, wend;

        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((batch_size * channel_size + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);

        cu_visit_im<T, decltype(visitor)><<<grid_size, block_size, 0, mpc::AbstractContext::_s_stream>>>(
            input_height, out_height, stride_height, padding_height, ksize_height,
            input_width, out_width, stride_width, padding_width, ksize_width,
            src, target, src_stride, target_stride, batch_size * channel_size, visitor);

}
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    mpc_pool2d, ops::MpcPoolKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    mpc_pool2d_grad, ops::MpcPoolGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
