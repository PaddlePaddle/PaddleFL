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

#pragma once

#include "Eigen/Core"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/framework/eigen.h"

#include "core/paddlefl_mpc/mpc_protocol/abstract_context.h"
#include "./tensor_adapter.h"

#define PFL_CUDA_THREAD_SIZE 512

namespace common {

template <typename T, typename Func>
static __global__ void cu_func(Func func, const T* lhs, const T* rhs, T* out, size_t size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    while (col < size) {
        out[col] = func(lhs[col], rhs[col]);
        col += blockDim.x * gridDim.x;
    }
}

template <typename T, typename Func, typename U = T>
static __global__ void cu_func(Func func, const U* lhs, T* out, size_t size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    while (col < size) {
        out[col] = func(lhs[col]);
        col += blockDim.x * gridDim.x;
    }
}

struct CudaMultParam {
    size_t batch_size;
    bool sum_reduce_batch;

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
static __global__ void cu_mult(const T* lhs, const T* rhs, T* out,
                               CudaMultParam param) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    using Eigen::Map;
    using Eigen::Matrix;
    using Eigen::RowMajor;
    using Eigen::Dynamic;
    using Eigen::Stride;

    if (param.sum_reduce_batch && idx >= param.out_row * param.out_col) {
        return;
    }

    while (idx < param.out_size) {
        size_t batch = idx / (param.out_row * param.out_col);
        size_t ij = idx % (param.out_row * param.out_col);
        size_t i = ij / param.out_col;
        size_t j = ij % param.out_col;

        Map<const Matrix<T, 1, Dynamic, RowMajor>, 0, Stride<0, Dynamic> > x(
            lhs + batch * param.x_batch_offset_base + param.x_offset_base * i,
            param.x_len, param.x_stride);

        Map<const Matrix<T, 1, Dynamic, RowMajor>, 0, Stride<0, Dynamic> > y(
            rhs + batch * param.y_batch_offset_base + param.y_offset_base * j,
            param.y_len, param.y_stride);

        // TODO: improve concurrency
        if (param.sum_reduce_batch) {
            out[idx % (param.out_row * param.out_col)] += x.dot(y);
            idx += param.out_row * param.out_col;
            continue;
        } else {
            out[idx] = x.dot(y);
        }

        idx += blockDim.x * gridDim.x;
    }

    return;
}

template <typename T>
class CudaPaddleTensor : public TensorAdapter<T> {
public:
    CudaPaddleTensor(const paddle::platform::DeviceContext *device_ctx)
        : _device_ctx(device_ctx), _scaling_factor(0) {
            _device_ctx = device_ctx;
        }

    CudaPaddleTensor(const paddle::platform::DeviceContext *device_ctx,
                     const paddle::framework::Tensor &src)
        : CudaPaddleTensor(_device_ctx) {
            // it seems that init list failed, weird
            _device_ctx = device_ctx;
            _tensor.ShareDataWith(src);
        }

    virtual ~CudaPaddleTensor() = default;

    T *data() override { return _tensor.data<T>(); }

    const T *data() const override { return _tensor.data<T>(); }

    const paddle::framework::Tensor* paddle_tensor() const {
        return &_tensor;
    }

    paddle::framework::Tensor* mutable_paddle_tensor() {
        return &_tensor;
    }

    std::vector<size_t> shape() const override {
        return paddle::framework::vectorize<size_t>(_tensor.dims());
    }

    size_t numel() const override { return _tensor.numel(); }

    void add(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op0, const T& op1) { return op0 + op1; }, rhs, ret);
    }

    void sum(TensorAdapter<T> *ret) const override {
        auto ret_ = dynamic_cast<CudaPaddleTensor<T> *>(ret);

        PADDLE_ENFORCE_EQ(1, ret_->_tensor.numel(),
                          "Result numel should be one.");

        auto eigen_x = paddle::framework::EigenVector<T>::Flatten(_tensor);
        auto eigen_z = paddle::framework::EigenVector<T>::Flatten(ret_->_tensor);

        auto &place = *eigen_device();

        eigen_z.device(place) = eigen_x.sum();
    }

    void copy(TensorAdapter<T>* ret) const override {
        run_on_cuda([] __device__ (const T& in) { return in; }, ret->data(), data(), numel(), device_ctx()->stream());
    }

    // TODO:
    void add128(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret,
                bool lhs_128, bool rhs_128) const override {
    }

    // TODO:
    void sub128(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret,
                bool lhs_128, bool rhs_128) const override {
    }

    // TODO:
    void mul128_with_truncate(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret,
                              bool lhs_128, bool rhs_128) const override {
    }

    void sub(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op0, const T& op1) { return op0 - op1; }, rhs, ret);
    }

    void negative(TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op) { return -op; }, ret);
    }

    void mul(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op0, const T& op1) { return op0 * op1; }, rhs, ret);
    }

    void div(const TensorAdapter<T> *rhs, TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op0, const T& op1) { return op0 / op1; }, rhs, ret);
    }

    void mat_mul(const TensorAdapter<T> *rhs,
                 TensorAdapter<T> *ret,
                 bool transpose_lhs = false,
                 bool transpose_rhs = false,
                 bool sum_reduce_batch = false) const override {

        auto rhs_ = dynamic_cast<const CudaPaddleTensor<T> *>(rhs);
        auto ret_ = dynamic_cast<CudaPaddleTensor<T> *>(ret);

        auto &mat_a = *this->paddle_tensor();
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

        PADDLE_ENFORCE(mat_a.dims()[rank_a - 1 - transpose_lhs]
                       == mat_b.dims()[rank_b - 2 + transpose_rhs],
                       "W_A != H_B.");

        auto batch_size = rank_a == 3 ? mat_a.dims()[0] : 1;
        auto batch_size_b = rank_b == 3 ? mat_b.dims()[0] : 1;

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

        param.out_row = transpose_lhs ? lhs_col : lhs_row;
        param.out_col = transpose_rhs ? rhs_row : rhs_col;
        param.out_size = batch_size * param.out_row * param.out_col;
        param.sum_reduce_batch = sum_reduce_batch;

        param.x_stride = Eigen::Stride<0, Eigen::Dynamic>(0, transpose_lhs ? lhs_col : 1);
        param.y_stride = Eigen::Stride<0, Eigen::Dynamic>(0, transpose_rhs ? 1 : rhs_col);

        param.x_len = transpose_lhs ? lhs_row : lhs_col;
        param.y_len = transpose_rhs ? rhs_col : rhs_row;

        param.x_offset_base = transpose_lhs ? 1 : lhs_col;
        param.y_offset_base = transpose_rhs ? rhs_col : 1;

        param.x_batch_offset_base = lhs_row * lhs_col;
        param.y_batch_offset_base = rhs_row * rhs_col * (batch_size_b!= 1);

        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((param.out_size + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);
        assign_to_tensor<T>(ret_, 0);
        cu_mult<T><<<grid_size, block_size, 0, device_ctx()->stream()>>>(data(), rhs_->data(), ret_->data(), param);
        // cudaStreamSynchronize(device_ctx()->stream());
    }


    void bitwise_xor(const TensorAdapter<T> *rhs,
                     TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op0, const T& op1) { return op0 ^ op1; }, rhs, ret);
    }

    void bitwise_and(const TensorAdapter<T> *rhs,
                     TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op0, const T& op1) { return op0 & op1; }, rhs, ret);
    }

    void bitwise_or(const TensorAdapter<T> *rhs,
                    TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op0, const T& op1) { return op0 | op1; }, rhs, ret);
    }


    void bitwise_not(TensorAdapter<T> *ret) const override {
        run_on_cuda([] __device__ (const T& op) { return ~op; }, ret);
    }


    void lshift(size_t rhs, TensorAdapter<T> *ret) const override {
        run_on_cuda([rhs] __device__ (const T& op) { return op << rhs; }, ret);
    }

    void rshift(size_t rhs, TensorAdapter<T> *ret) const override {
        run_on_cuda([rhs] __device__ (const T& op) { return op >> rhs; }, ret);
    }


    void logical_rshift(size_t rhs, TensorAdapter<T> *ret) const override {
        auto logical_rshift_functor = [rhs] __device__ (const T& lhs) -> T {
            const size_t word_len = sizeof(T) * 8;
            T mask = (T)1 << word_len - rhs - 1;
            mask |= mask - 1;
            mask = rhs >= word_len ? 0 : mask;
            return lhs >> rhs & mask;
        };

        run_on_cuda(logical_rshift_functor, ret->data(), data(),
                    numel(), device_ctx()->stream());
    }

    paddle::framework::Tensor &tensor() { return _tensor; }

    const paddle::framework::Tensor &tensor() const { return _tensor; }

    void reshape(const std::vector<size_t> &shape) {
        std::vector<int64_t> shape_(shape.cbegin(), shape.cend());
        paddle::framework::DDim dim(shape_.data(), shape_.size());
        _tensor.mutable_data<T>(dim, place(), 0);
    }

    const paddle::platform::CUDADeviceContext* device_ctx() const {
        return dynamic_cast<const paddle::platform::CUDADeviceContext*>(_device_ctx);
    }

    size_t scaling_factor() const override { return _scaling_factor; }

    size_t &scaling_factor() override { return _scaling_factor; }

    void slice(size_t begin_idx, size_t end_idx,
               TensorAdapter<T> *ret) const override {
        auto ret_ = dynamic_cast<CudaPaddleTensor<T> *>(ret);
        ret_->_tensor = _tensor.Slice(begin_idx, end_idx);

        ret->scaling_factor() = scaling_factor();
    }

    template <typename U>
    CudaPaddleTensor& from_float_point_type(const paddle::framework::Tensor &tensor,
                                            size_t scaling_factor) {

        double scale = std::pow(2, scaling_factor);

        auto cast = [scale] __device__ (U a) -> T { return a * scale; };

        _tensor.mutable_data<T>(tensor.dims(), place(), 0);

        run_on_cuda(cast, data(), tensor.data<U>(), numel(), device_ctx()->stream());
        this->scaling_factor() = scaling_factor;

        return *this;
    }

    template <typename U>
    CudaPaddleTensor &from_float_point_scalar(const U &scalar,
                                              const std::vector<size_t> &shape,
                                              size_t scaling_factor) {
        double scale = std::pow(2, scaling_factor);

        auto trans = [scale, scalar] __device__ (T) -> T { return scalar * scale; };

        reshape(shape);

        run_on_cuda(trans, data(), data(), numel(), device_ctx()->stream());

        this->scaling_factor() = scaling_factor;

        return *this;
    }

    template<int Rank>
    void Transpose(const std::vector<int> axis, TensorAdapter<T>* ret) const {
        // copied from "paddle/fluid/operators/math/math_function.h"
        // error occurs when includes from paddle
        Eigen::array<int, Rank> permute;
        for (int i = 0; i < Rank; i++) {
            permute[i] = axis[i];
        }
        auto eigen_in = paddle::framework::EigenTensor<T, Rank>::From(_tensor);
        auto eigen_out = paddle::framework::EigenTensor<T, Rank>::From(*dynamic_cast<CudaPaddleTensor<T>*>(ret)->mutable_paddle_tensor());
        eigen_out.device(*eigen_device()) = eigen_in.shuffle(permute);
    }

    template<int Rank>
    void Broadcast(const std::vector<int> axis, TensorAdapter<T>* ret) const {
        // input, output and axis rank = Rank
        // TODO: arbitrary ranks
        auto in = paddle::framework::EigenTensor<T, Rank>::From(_tensor);
        auto out = paddle::framework::EigenTensor<T, Rank>::From(dynamic_cast<CudaPaddleTensor<T>*>(ret)->_tensor);

        Eigen::array<int, Rank> bcast;
        for (int i = 0; i < Rank; i++) {
            bcast[i] = axis[i];
        }

        auto &place = *eigen_device();
        out.device(place) = in.broadcast(bcast);
    }

    template<int Rank>
    void SumReduceLastDim(TensorAdapter<T>* ret) const {
        using namespace paddle::framework;
        auto in = EigenTensor<T, Rank>::From(_tensor);

        Eigen::array<int, 1> axis({ Rank - 1 });
        int newshape[Rank - 1];
        for (int i = 0; i < Rank - 1; ++i) {
            newshape[i] = _tensor.dims()[i];
        }
        auto out = EigenTensor<T, Rank - 1>::From(
            dynamic_cast<CudaPaddleTensor<T>*>(ret)->_tensor, DDim(newshape, Rank - 1));

        auto &place = *eigen_device();
        out.device(place) = in.sum(axis);

    }

    void sum_reduce_last_dim(TensorAdapter<T>* ret) const {
        auto in_dims_size = shape().size();
        switch (in_dims_size) {
        case 1:
            SumReduceLastDim<1>(ret);
            break;
        case 2:
            SumReduceLastDim<2>(ret);
            break;
        case 3:
            SumReduceLastDim<3>(ret);
            break;
        case 4:
            SumReduceLastDim<4>(ret);
            break;
        case 5:
            SumReduceLastDim<5>(ret);
            break;
        default:
            throw std::invalid_argument("unsupported input dim size: " + std::to_string(in_dims_size));
        }
    }

    // slice by shape[0] of index
    // e.g. x.shape = [2, 3, 4]
    //      data of x[1] = x.slice(1, 2)
    //      x[1]->shape = [3, 4]
    std::shared_ptr<TensorAdapter<T>> operator[](size_t index) {
        PADDLE_ENFORCE_GT(this->shape().size(), 1,
                          "lhs's shape must great than 1.");
        auto slice_shape = this->shape();
        slice_shape.erase(slice_shape.begin());
        std::shared_ptr<CudaPaddleTensor<T>> ret = std::make_shared<CudaPaddleTensor<T>>(_device_ctx);
        ret->reshape(slice_shape);

        this->slice(index, index + 1, ret.get());
        ret->reshape(slice_shape);
        return ret;
    }

    const std::shared_ptr<TensorAdapter<T>> operator[](size_t index) const {
        return const_cast<CudaPaddleTensor*>(this)->operator[](index);
    }

private:
    paddle::platform::Place place() const { return _device_ctx->GetPlace(); }

    Eigen::GpuDevice* eigen_device() const {
        return dynamic_cast<const paddle::platform::CUDADeviceContext *>(_device_ctx)
            ->eigen_device();
    }

    template<typename Func>
    void run_on_cuda(Func func, const TensorAdapter<T> *rhs,
                     TensorAdapter<T> *ret) const {

        auto rhs_ = dynamic_cast<const CudaPaddleTensor<T> *>(rhs);
        auto ret_ = dynamic_cast<CudaPaddleTensor<T> *>(ret);

        PADDLE_ENFORCE_EQ(_tensor.dims(), rhs_->_tensor.dims(),
                          "Input dims should be equal.");

        auto size = numel();

        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((size + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);

        cu_func<T><<<grid_size, block_size, 0, device_ctx()->stream()>>>(
            func, data(), rhs_->data(), ret_->data(), size);

        // cudaStreamSynchronize(device_ctx()->stream());
    };

    template<typename Func, typename U>
    static void run_on_cuda(Func func, T* dst, const U* src,
                            size_t size, cudaStream_t stream) {
        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((size + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);
        cu_func<T><<<grid_size, block_size, 0, stream>>>(func, src, dst, size);

        // cudaStreamSynchronize(stream);
    }

    template<typename Func>
    void run_on_cuda(Func func, TensorAdapter<T>* ret) const {
        run_on_cuda(func, ret->data(), data(), numel(), device_ctx()->stream());
    }

private:
    paddle::framework::Tensor _tensor;

    const paddle::platform::DeviceContext *_device_ctx;

    size_t _scaling_factor;

};

template<typename T>
inline void assign_to_tensor(TensorAdapter<T>* input, T assign_num) {
        cudaStream_t stream = paddle::mpc::AbstractContext::_s_stream;
        dim3 block_size = dim3(PFL_CUDA_THREAD_SIZE, 1);
        dim3 grid_size = dim3((input->numel() + PFL_CUDA_THREAD_SIZE - 1) / PFL_CUDA_THREAD_SIZE, 1);
        cu_func<T><<<grid_size, block_size, 0, stream>>>(
            [assign_num] __device__ (const T& in) { return assign_num; },
            input->data(), input->data(), input->numel());
        // cudaStreamSynchronize(stream);
}

} // namespace common
