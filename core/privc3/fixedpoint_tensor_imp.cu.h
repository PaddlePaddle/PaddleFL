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

namespace aby3 {

#ifdef __NVCC__
// reduce last dim
template <typename T, size_t N>
void FixedPointTensor<T, N>::reduce(const FixedPointTensor<T, N>* input,
                                    FixedPointTensor<T, N>* ret) {
    //enfoce shape: input->shape[0 ... (n-2)] == ret shape
    dynamic_cast<const common::CudaPaddleTensor<T>*>(input->_share[0])->sum_reduce_last_dim(ret->_share[0]);
    dynamic_cast<const common::CudaPaddleTensor<T>*>(input->_share[1])->sum_reduce_last_dim(ret->_share[1]);
}

template< typename T, size_t N>
void FixedPointTensor<T, N>::softmax(FixedPointTensor<T, N>* ret,
                                     bool use_relu, bool use_long_div) const {
    // softmax axis = -1
    const size_t col = *(shape().end() - 1);
    const size_t row = numel() / col;

    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    // 11 for allocating temp tensor
    for (size_t i = 0; i < 11; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }

    temp[0]->reshape({row, col});
    temp[1]->reshape({row, col});
    FixedPointTensor<T, N> x(temp[0].get(), temp[1].get());

    if (!use_relu) {
        temp[2]->reshape({col, row});
        temp[3]->reshape({col, row});

        temp[4]->reshape({1, row});
        temp[5]->reshape({1, row});
    }
    FixedPointTensor<T, N> x_t(temp[2].get(), temp[3].get());
    FixedPointTensor<T, N> max_x_t(temp[4].get(), temp[5].get());

    temp[6]->reshape({row, 1});
    temp[7]->reshape({row, 1});
    FixedPointTensor<T, N> max_x(temp[6].get(), temp[7].get());

    temp[8]->reshape({row, col});
    temp[9]->reshape({row, col});
    FixedPointTensor<T, N> max_x_broadcast(temp[8].get(), temp[9].get());

    temp[10]->reshape({row, col});
    auto exp_lower_bound = temp[10].get();

    auto transpose = [](const TensorAdapter<T>* in, TensorAdapter<T>* out) {
        std::vector<int> axis{ 1, 0 };
        // suppose input dims = 2
        dynamic_cast<const common::CudaPaddleTensor<T>*>(in)->template Transpose<2>(axis, out);
    };

    auto broadcast = [](const TensorAdapter<T>* in, TensorAdapter<T>* out) {
        // suppose input dims = 2
        const int col = out->shape()[1];
        std::vector<int> axis{ 1, col };
        dynamic_cast<const common::CudaPaddleTensor<T>*>(in)->template Broadcast<2>(axis, out);
    };

    share(0)->copy(x.mutable_share(0));
    share(1)->copy(x.mutable_share(1));

    if (use_relu) {

        x.relu(&x);

    } else { // use exp
        transpose(x.share(0), x_t.mutable_share(0));
        transpose(x.share(1), x_t.mutable_share(1));

        // x = max(input - max(input), exp_lower_bound)
        x_t.max_pooling(&max_x_t);

        transpose(max_x_t.share(0), max_x.mutable_share(0));
        transpose(max_x_t.share(1), max_x.mutable_share(1));

        broadcast(max_x.share(0), max_x_broadcast.mutable_share(0));
        broadcast(max_x.share(1), max_x_broadcast.mutable_share(1));

        x.sub(&max_x_broadcast, &x);

        // n = 64, see exp
        assign_to_tensor(exp_lower_bound, (T)(-64 * (1 << N)));
        exp_lower_bound->scaling_factor() = N;

        x.sub(exp_lower_bound, &x);
        x.relu(&x);
        x.add(exp_lower_bound, &x);

        x.exp(&x);
    }

    // reuse max_x as sum
    reduce(&x, &max_x);

    if (!use_long_div) { // invert sum by Newton's method
    // divisor range = [1/col, 1.0]
    // TODO: find better iter num & init val
        reciprocal(&max_x, &max_x, 16, 0.5 / col);
    }

    broadcast(max_x.share(0), max_x_broadcast.mutable_share(0));
    broadcast(max_x.share(1), max_x_broadcast.mutable_share(1));

    if (use_long_div) {
        x.long_div(&max_x_broadcast, &x, 1);
    } else {
        x.mul(&max_x_broadcast, &x);
    }

    x.share(0)->copy(ret->mutable_share(0));
    x.share(1)->copy(ret->mutable_share(1));
}

template<typename T, size_t N>
void FixedPointTensor<T, N>::calc_precision_recall(
    const FixedPointTensor* tp_fp_fn,
    TensorAdapter<T>* ret) {
    PADDLE_ENFORCE_EQ(tp_fp_fn->shape().size(), 1,
                      "multi-classification not support yet");

    PADDLE_ENFORCE_EQ(tp_fp_fn->shape()[0], 3,
                      "store tp fp fn for binary-classification only");

    PADDLE_ENFORCE_EQ(ret->shape().size(), 1,
                      "multi-classification not support yet");

    PADDLE_ENFORCE_EQ(ret->shape()[0], 3,
                      "store precision recall f1-score"
                      "for binary-classification only");
    // 5 for allocating temp tensor
    std::vector<std::shared_ptr<TensorAdapter<T>>> temp;
    for (size_t i = 0; i < 7; ++i) {
        temp.emplace_back(
            tensor_factory()->template create<T>());
    }
    std::vector<size_t> shape_ = {3};

    std::vector<size_t> shape_one = {1};

    temp[0]->reshape(shape_one);
    temp[1]->reshape(shape_one);
    FixedPointTensor scalar(temp[0].get(), temp[1].get());

    temp[2]->reshape(shape_one);
    temp[3]->reshape(shape_one);
    FixedPointTensor scalar2(temp[2].get(), temp[3].get());

    temp[4]->reshape(shape_one);
    temp[5]->reshape(shape_one);
    FixedPointTensor tmp_(temp[4].get(), temp[5].get());

    auto get = [&tp_fp_fn](size_t idx, FixedPointTensor* dest) {
        tp_fp_fn->share(0)->slice(idx, idx + 1, dest->mutable_share(0));
        tp_fp_fn->share(1)->slice(idx, idx + 1, dest->mutable_share(1));
    };

    get(0, &scalar);
    get(1, &scalar2);

    // tp + fp
    scalar.add(&scalar2, &tmp_);

    scalar.long_div(&tmp_, &tmp_);

    temp[6]->reshape(shape_one);

    tmp_.reveal(temp[6].get());

    T buf[3];
    cudaMemcpy(buf, temp[6]->data(), sizeof(T), cudaMemcpyDeviceToHost);

    get(2, &scalar2);

    // tp + fn
    scalar.add(&scalar2, &tmp_);

    scalar.long_div(&tmp_, &tmp_);
    tmp_.reveal(temp[6].get());

    cudaMemcpy(buf + 1, temp[6]->data(), sizeof(T), cudaMemcpyDeviceToHost);

    float precision = 1.0 * buf[0] / (T(1) << N);
    float recall = 1.0 * buf[1] / (T(1) << N);
    float f1_score = 0.0;
    if (precision + recall > 0) {
        f1_score = 2 * precision * recall / (precision + recall);
    }

    buf[2] = T(f1_score * (T(1) << N));

    ret->scaling_factor() = N;
    cudaMemcpy(ret->data(), buf, sizeof(buf), cudaMemcpyHostToDevice);
}
#endif // __NVCC__

} // namespace aby3
