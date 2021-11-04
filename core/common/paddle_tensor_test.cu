// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle_tensor_impl.cu.h"

#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace common {

using namespace paddle::framework;

TEST(CudaPaddleTensorTest, ctor) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    CudaPaddleTensor<int64_t> ct(cuda_ctx);
  }
  {
    Tensor tensor;
    int64_t* buf = tensor.mutable_data<int64_t>({3}, gpu);
    CudaPaddleTensor<int64_t> ct(cuda_ctx, tensor);
    EXPECT_EQ(3, ct.numel());
    auto shape = ct.shape();
    EXPECT_EQ(1, shape.size());
    EXPECT_EQ(3, shape[0]);
  }
}

template <typename T>
static __global__ void fill(T* buf, size_t s = 3, int offset = 1) {
    for (int i = 0; i < s; ++i) {
        buf[i] = i + offset;
    }
}

TEST(CudaPaddleTensorTest, data) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor tensor;
    int64_t* buf = tensor.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf);
    cuda_ctx->Wait();
    CudaPaddleTensor<int64_t> ct(cuda_ctx, tensor);
    int64_t host[3];
    cudaMemcpy(host, ct.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(1, host[0]);
    EXPECT_EQ(2, host[1]);
    EXPECT_EQ(3, host[2]);
  }
}

TEST(CudaPaddleTensorTest, add) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, out);
    ct0.add(&ct0, &ct1);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct1.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(2, host[0]);
    EXPECT_EQ(4, host[1]);
    EXPECT_EQ(6, host[2]);
  }
}

TEST(CudaPaddleTensorTest, sub) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor in1;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    int64_t* buf1 = in1.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 3, 0);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf1, 3, 2);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, in1);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    ct0.sub(&ct1, &ct2);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct2.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(-2, host[0]);
    EXPECT_EQ(-2, host[1]);
    EXPECT_EQ(-2, host[2]);
  }
}

TEST(CudaPaddleTensorTest, neg) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, out);
    ct0.negative(&ct1);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct1.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(-1, host[0]);
    EXPECT_EQ(-2, host[1]);
    EXPECT_EQ(-3, host[2]);
  }
}

TEST(CudaPaddleTensorTest, mul) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, out);
    ct0.mul(&ct0, &ct1);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct1.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(1, host[0]);
    EXPECT_EQ(4, host[1]);
    EXPECT_EQ(9, host[2]);
  }
}

TEST(CudaPaddleTensorTest, div) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor in1;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    int64_t* buf1 = in1.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 3, 3);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf1, 3, 1);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, in1);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    ct0.div(&ct1, &ct2);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct2.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(3, host[0]);
    EXPECT_EQ(2, host[1]);
    EXPECT_EQ(1, host[2]);
  }
}

TEST(CudaPaddleTensorTest, mat_mul) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor in1;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({2, 2}, gpu);
    int64_t* buf1 = in1.mutable_data<int64_t>({2, 2}, gpu);
    out.mutable_data<int64_t>({2, 2}, gpu);

    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 4, 1);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf1, 4, 2);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, in1);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    ct0.mat_mul(&ct1, &ct2);
    cuda_ctx->Wait();

    int64_t host[4];
    cudaMemcpy(host, ct2.data(), 4 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(10, host[0]);
    EXPECT_EQ(13, host[1]);
    EXPECT_EQ(22, host[2]);
    EXPECT_EQ(29, host[3]);
  }
}

TEST(CudaPaddleTensorTest, mat_mul_sum_reduce_batch) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor in1;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({2, 2, 3}, gpu);
    int64_t* buf1 = in1.mutable_data<int64_t>({2, 3, 2}, gpu);
    out.mutable_data<int64_t>({2, 2}, gpu);

    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 12, 0);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf1, 12, 0);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, in1);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    ct0.mat_mul(&ct1, &ct2, 0, 0, 1);
    cuda_ctx->Wait();

    int64_t host[4];
    cudaMemcpy(host, ct2.data(), 4 * sizeof(int64_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(182, host[0]);
    EXPECT_EQ(206, host[1]);
    EXPECT_EQ(272, host[2]);
    EXPECT_EQ(314, host[3]);

  }
}

TEST(CudaPaddleTensorTest, xor) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor in1;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    int64_t* buf1 = in1.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 3, 0);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf1, 3, 2);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, in1);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    ct0.bitwise_xor(&ct1, &ct2);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct2.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(2, host[0]);
    EXPECT_EQ(2, host[1]);
    EXPECT_EQ(6, host[2]);
  }
}

TEST(CudaPaddleTensorTest, and) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor in1;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    int64_t* buf1 = in1.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 3, 0);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf1, 3, 2);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, in1);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    ct0.bitwise_and(&ct1, &ct2);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct2.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, host[0]);
    EXPECT_EQ(1, host[1]);
    EXPECT_EQ(0, host[2]);
  }
}

TEST(CudaPaddleTensorTest, or) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor in1;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    int64_t* buf1 = in1.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 3, 0);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf1, 3, 2);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, in1);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    ct0.bitwise_or(&ct1, &ct2);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct2.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(2, host[0]);
    EXPECT_EQ(3, host[1]);
    EXPECT_EQ(6, host[2]);
  }
}

TEST(CudaPaddleTensorTest, lshift) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, out);
    ct0.lshift(1, &ct1);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct1.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(2, host[0]);
    EXPECT_EQ(4, host[1]);
    EXPECT_EQ(6, host[2]);
  }
}

TEST(CudaPaddleTensorTest, rshift) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, out);
    ct0.rshift(1, &ct1);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct1.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, host[0]);
    EXPECT_EQ(1, host[1]);
    EXPECT_EQ(1, host[2]);
  }
}

TEST(CudaPaddleTensorTest, logical_rshift) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({3}, gpu);
    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf);

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct1(cuda_ctx, out);
    ct0.logical_rshift(1, &ct1);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct1.data(), 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, host[0]);
    EXPECT_EQ(1, host[1]);
    EXPECT_EQ(1, host[2]);
  }
}

TEST(CudaPaddleTensorTest, trans) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    int64_t* buf = in.mutable_data<int64_t>({2, 2}, gpu);
    out.mutable_data<int64_t>({2, 2}, gpu);

    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 4, 1);

    CudaPaddleTensor<int64_t> ct0(cuda_ctx, in);
    CudaPaddleTensor<int64_t> ct2(cuda_ctx, out);
    std::vector<int> axis = { 1, 0 };
    ct0.Transpose<2>(axis, &ct2);
    cuda_ctx->Wait();

    int64_t host[4];
    cudaMemcpy(host, ct2.data(), 4 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(1, host[0]);
    EXPECT_EQ(3, host[1]);
    EXPECT_EQ(2, host[2]);
    EXPECT_EQ(4, host[3]);
  }
}

TEST(CudaPaddleTensorTest, from_fp_tensor) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor in;
    Tensor out;

    float* buf = in.mutable_data<float>({3}, gpu);
    out.mutable_data<int64_t>({3}, gpu);

    fill<<<1, 1, 0, cuda_ctx->stream()>>>(buf, 3, 1);

    CudaPaddleTensor<int64_t> ct(cuda_ctx, out);
    ct.from_float_point_type<float>(in, 2);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct.data(), 4 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(4, host[0]);
    EXPECT_EQ(8, host[1]);
    EXPECT_EQ(12, host[2]);
  }
}

TEST(CudaPaddleTensorTest, from_fp_val) {
  paddle::platform::CUDAPlace gpu(0);
  paddle::platform::DeviceContextPool::Init({gpu});
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto cuda_ctx = pool.GetByPlace(gpu);
  {
    Tensor out;

    out.mutable_data<int64_t>({3}, gpu);

    CudaPaddleTensor<int64_t> ct(cuda_ctx, out);
    ct.from_float_point_scalar<float>(2.0, { 3 }, 2);
    cuda_ctx->Wait();

    int64_t host[3];
    cudaMemcpy(host, ct.data(), 4 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(8, host[0]);
    EXPECT_EQ(8, host[1]);
    EXPECT_EQ(8, host[2]);
  }
}
}
