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

#include <stdio.h>

#include "Eigen/Core"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "unsupported/Eigen/CXX11/Tensor"

#include "./functor.h"
#include "./tensor_adapter.h"

namespace common {

template <typename T>
class PaddleTensor;

struct CudaMultParam {
    size_t batch_size;

    size_t out_row;
    size_t out_col;
    size_t out_size;

    size_t x_len;
    size_t y_len;

    size_t x_offset_base;
    size_t y_offset_base;

    size_t x_batch_offset_base;
    size_t y_batch_offset_base;

    Eigen::Stride<0, Eigen::Dynamic> x_stride;
    Eigen::Stride<0, Eigen::Dynamic> y_stride;
};

template <typename T>
__global__ void cu_mult(T *lhs, T *rhs, T *out,
                        CudaMultParam param) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    using Eigen::Map;
    using Eigen::Matrix;
    using Eigen::RowMajor;
    using Eigen::Dynamic;
    using Eigen::Stride;

    while (idx < param.out_size)
    {
        size_t batch = idx / (param.out_row * param.out_col);
        size_t ij = idx % (param.out_row * param.out_col);
        size_t i = ij / param.out_col;
        size_t j = ij % param.out_col;

        Map<Matrix<T, 1, Dynamic, RowMajor>, 0, Stride<0, Dynamic> > x(
            lhs + batch * param.x_batch_offset_base + param.x_offset_base * i,
            param.x_len, param.x_stride);

        Map<Matrix<T, 1, Dynamic, RowMajor>, 0, Stride<0, Dynamic> > y(
            rhs + batch * param.y_batch_offset_base + param.y_offset_base * j,
            param.y_len, param.y_stride);

        out[idx] = x.dot(y);

        idx += blockDim.x * gridDim.x;
    }

    return;
}

template <typename T>
void MatMul<T>::mat_mul(const TensorAdapter<T>* lhs,
                 const TensorAdapter<T>* rhs,
                 TensorAdapter<T>* ret,
                 bool trans_lhs,
                 bool trans_rhs) {
  auto lhs_ = dynamic_cast<const PaddleTensor<T> *>(lhs);
  auto rhs_ = dynamic_cast<const PaddleTensor<T> *>(rhs);
  auto ret_ = dynamic_cast<PaddleTensor<T> *>(ret);

  auto &mat_a = *lhs_->paddle_tensor();
  auto &mat_b = *rhs_->paddle_tensor();
  auto &mat_out = *ret_->paddle_tensor();

  // tensor with dims like [ h, w ] or [ batch_size , h, w ] is matrix
  auto is_matrix = [](const paddle::framework::Tensor &t) -> bool {
    return t.dims().size() == 2 || t.dims().size() == 3;
  };

  // PADDLE_ENFORCE(mat_a.place() == mat_b.place() &&
  //                    mat_a.place() == mat_out.place(),
  //                "The places of matrices must be same");

  PADDLE_ENFORCE(is_matrix(mat_a) && is_matrix(mat_b) && is_matrix(mat_out),
                 "The input and output of matmul must be matrix "
                 "or batched matrix.");

  PADDLE_ENFORCE(mat_a.dims().size() >= mat_b.dims().size(),
                 "Only following dims are supported: "
                 "Mat A is [BatchSize, H, W] and Mat B is [BatchSize, H, W]."
                 "Mat A is [BatchSize, H, W] and Mat B is [H, W]."
                 "Mat A is [H, W] and Mat B is [H, W].");

  size_t rank_a = mat_a.dims().size();
  size_t rank_b = mat_b.dims().size();

  PADDLE_ENFORCE(mat_a.dims()[rank_a - 1 - trans_lhs]
                 == mat_b.dims()[rank_b - 2 + trans_rhs],
                 "W_A != H_B.");

  auto batch_size = rank_a == 3 ? mat_a.dims()[0] : 1; auto batch_size_b = rank_b == 3 ? mat_b.dims()[0] : 1;

  PADDLE_ENFORCE(batch_size_b == batch_size || batch_size_b == 1,
                 "Mat B BatchSize mismatched.");

  // PADDLE_ENFORCE(t_c.dimension(0) == batch_size,
  //                "Result Mat BatchSize mismatched.");

  struct CudaMultParam param;

  param.batch_size = batch_size;

  size_t lhs_row = mat_a.dims()[rank_a - 2];
  size_t lhs_col = mat_a.dims()[rank_a - 1];
  size_t rhs_row = mat_b.dims()[rank_b - 2];
  size_t rhs_col = mat_b.dims()[rank_b - 1];

  param.out_row = trans_lhs ? lhs_col : lhs_row;
  param.out_col = trans_rhs ? rhs_row : rhs_col;
  param.out_size = batch_size * param.out_row * param.out_col;

  param.x_stride = Eigen::Stride<0, Eigen::Dynamic>(0, trans_lhs ? lhs_col : 1);
  param.y_stride = Eigen::Stride<0, Eigen::Dynamic>(0, trans_rhs ? 1 : rhs_col);

  param.x_len = trans_lhs ? lhs_row : lhs_col;
  param.y_len = trans_rhs ? rhs_col : rhs_row;

  param.x_offset_base = trans_lhs ? 1 : lhs_col;
  param.y_offset_base = trans_rhs ? rhs_col : 1;

  param.x_batch_offset_base = lhs_row * lhs_col;
  param.y_batch_offset_base = rhs_row * rhs_col * (batch_size_b!= 1);

  T* dev_lhs;
  T* dev_rhs;
  T* dev_ret;

  cudaMalloc((void **)&dev_lhs, sizeof(T) * mat_a.numel());
  cudaMalloc((void **)&dev_rhs, sizeof(T) * mat_b.numel());
  cudaMalloc((void **)&dev_ret, sizeof(T) * mat_out.numel());
  cudaMemcpy(dev_lhs, lhs->data(), sizeof(T) * mat_a.numel(), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_rhs, rhs->data(), sizeof(T) * mat_b.numel(), cudaMemcpyHostToDevice);

  // cu_mult<T><<<(param.out_size + 1023)/1024, 1024>>>(dev_lhs, dev_rhs, dev_ret, param);
  cu_mult<T><<<(param.out_size + 127)/128, 128>>>(dev_lhs, dev_rhs, dev_ret, param);
  // cu_mult<T><<<1, 1>>>(dev_lhs, dev_rhs, dev_ret, param);
  // cudaDeviceSynchronize();

  cudaMemcpy(ret->data(), dev_ret, sizeof(T) * mat_out.numel(), cudaMemcpyDeviceToHost);
  // cudaError_t err = cudaGetLastError();  // add
  // if (err != cudaSuccess)
  //     std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

  cudaFree(dev_lhs);
  cudaFree(dev_rhs);
  cudaFree(dev_ret);
}

} // namespace common
