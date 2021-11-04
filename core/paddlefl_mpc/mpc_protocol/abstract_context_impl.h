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

#ifdef USE_CUDA
#ifdef __NVCC__

#include "cuda_runtime.h"

namespace paddle {
namespace mpc {

using PRNG = common::PRNG;

// generate random from prng[0] or prng[1]
// @param next: use bool type for idx 0 or 1
template <typename T, template <typename> class Tensor>
void AbstractContext::gen_random(Tensor<T> &tensor, bool next) {
    get_prng(next).get_array(tensor.data(),
                             sizeof(T) * tensor.numel(),
                             // TODO: pass cuda stream
                             // tensor.device_ctx()->stream());
                             _s_stream);
}

template <typename T, template <typename> class Tensor>
void AbstractContext::gen_random_private(Tensor<T> &tensor) {
    get_prng(2).get_array(tensor.data(),
                          sizeof(T) * tensor.numel(), _s_stream);
}

template <typename T>
struct SubRandomTensor {
    void operator()(PRNG& prng, void* data, size_t len, cudaStream_t stream);
};

template <>
struct SubRandomTensor<int64_t> {
    void operator()(PRNG& prng, void* data, size_t len, cudaStream_t stream) {
        prng.array_sub64(data, len, stream);
    }
};

template <typename T, template <typename> class Tensor>
void AbstractContext::gen_zero_sharing_arithmetic(Tensor<T> &tensor) {
    get_prng(0).get_array(tensor.data(),
                          sizeof(T) * tensor.numel(), _s_stream);

    auto functor = SubRandomTensor<T>();
    functor(get_prng(1), tensor.data(),
            sizeof(T) * tensor.numel(), _s_stream);

}

template <typename T, template <typename> class Tensor>
void AbstractContext::gen_zero_sharing_boolean(Tensor<T> &tensor) {
    get_prng(0).get_array(tensor.data(),
                          sizeof(T) * tensor.numel(), _s_stream);

    get_prng(1).xor_array(tensor.data(),
                          sizeof(T) * tensor.numel(), _s_stream);
}

} // namespace mpc
} //namespace paddle
#endif // __NVCC__

#else // USE_CUDA

namespace paddle {
namespace mpc {

// generate random from prng[0] or prng[1]
// @param next: use bool type for idx 0 or 1
template <typename T> T AbstractContext::gen_random(bool next) {
    return get_prng(next).get<T>();
}

template <typename T, template <typename> class Tensor>
void AbstractContext::gen_random(Tensor<T> &tensor, bool next) {
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this, next](T &val) { val = this->template gen_random<T>(next); });
}

template <typename T>
T AbstractContext::gen_random_private() { return get_prng(2).get<T>(); }

template <typename T, template <typename> class Tensor>
void AbstractContext::gen_random_private(Tensor<T> &tensor) {
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this](T &val) { val = this->template gen_random_private<T>(); });
}

template <typename T>
T AbstractContext::gen_zero_sharing_arithmetic() {
    return get_prng(0).get<T>() - get_prng(1).get<T>();
}

template <typename T, template <typename> class Tensor>
void AbstractContext::gen_zero_sharing_arithmetic(Tensor<T> &tensor) {
    std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                  [this](T &val) {
                  val = this->template gen_zero_sharing_arithmetic<T>();
                  });
}

template <typename T>
T AbstractContext::gen_zero_sharing_boolean() {
    return get_prng(0).get<T>() ^ get_prng(1).get<T>();
}

template <typename T, template <typename> class Tensor>
void AbstractContext::gen_zero_sharing_boolean(Tensor<T> &tensor) {
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this](T &val) { val = this->template gen_zero_sharing_boolean<T>(); });
}

} // namespace mpc
} //namespace paddle

#endif // USE_CUDA
