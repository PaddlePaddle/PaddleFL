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

#include <cstddef>
#include <vector>

namespace paddle {
namespace mpc {

class AbstractNetwork {
public:
  AbstractNetwork() = default;

  virtual ~AbstractNetwork() = default;

  virtual void send(size_t party, const void *data, size_t size) = 0;

  virtual void recv(size_t party, void *data, size_t size) = 0;

  virtual void broadcast(const void *data, size_t size) {
    for (size_t i = 0; i < party_num(); ++i) {
      if (i == party_id()) {
        continue;
      }
      send(i, data, size);
    }
  }

  virtual void gather(void *data[], size_t size) {
    for (size_t i = 0; i < party_num(); ++i) {
      if (i == party_id()) {
        continue;
      }
      recv(i, data[i], size);
    }
  }

  template <typename T> void send(size_t party, const T &data) {
    send(party, &data, sizeof(T));
  }

  template <typename T, template <typename> class Tensor>
  void send(size_t party, const Tensor<T> &tensor) {
    send(party, tensor.data(), sizeof(T) * tensor.numel());
  }

  template <typename T> T recv(size_t party) {
    T ret;
    recv(party, &ret, sizeof(T));
    return ret;
  }

  template <typename T, template <typename> class Tensor>
  Tensor<T> &recv(size_t party, Tensor<T> &tensor) {
    recv(party, tensor.data(), sizeof(T) * tensor.numel());
    return tensor;
  }

  template <typename T> void broadcast(const T &data) {
    broadcast(&data, sizeof(T));
  }

  template <typename T> std::vector<T> gather() {
    std::vector<T> ret(party_num());
    for (size_t i = 0; i < party_num(); ++i) {
      if (i == party_id()) {
        continue;
      }
      recv(i, &ret[i], sizeof(T));
    }
    return ret;
  }

  template <typename T> void send(size_t party, const T *begin, const T *end) {
    send(party, begin, (end - begin) * sizeof(T));
  }

  template <typename T> T *recv(size_t party, T *begin, T *end) {
    recv(party, begin, (end - begin) * sizeof(T));
    return begin;
  }

  template <typename T>
  void broadcast(size_t party, const T *begin, const T *end) {
    broadcast(begin, (end - begin) * sizeof(T));
  }

  template <typename T> void gather(T *begin[], T *end[]) {
    gather(begin, sizeof(T) * (end[0] - begin[0]));
  }

  virtual size_t party_id() const = 0;

  virtual size_t party_num() const = 0;
};

} // namespace mpc
} // namespace paddle
