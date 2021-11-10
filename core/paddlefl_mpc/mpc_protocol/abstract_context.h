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

#include <algorithm>
#include <memory>

#include "core/paddlefl_mpc/mpc_protocol/abstract_network.h"
#ifdef USE_CUDA
#include "core/common/prng.cu.h"
#else
#include "core/common/prng.h"
#endif // USE_CUDA
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

namespace mpc {

using block = common::block;
using PseudorandomNumberGenerator = common::PseudorandomNumberGenerator;

class AbstractContext {
public:
  AbstractContext(size_t party, std::shared_ptr<AbstractNetwork> network) {
    set_party(party);
    set_network(network);
  };
  AbstractContext(const AbstractContext &other) = delete;

  AbstractContext &operator=(const AbstractContext &other) = delete;

  void set_party(size_t party) {
    _party = party;
  }

  void set_num_party(size_t num_party) {
    _num_party = num_party;
  }

  void set_network(std::shared_ptr<AbstractNetwork> network) {
    _network = network;
  }

  AbstractNetwork *network() { return _network.get(); }

  void set_random_seed(const block &seed, size_t idx);

  size_t party() const { return _party; }

  size_t pre_party() const {
    return (_party + _num_party - 1) % _num_party;
  }

  size_t next_party() const {
    return (_party + 1) % _num_party;
  }

  // generate random from prng[0] or prng[1]
  // @param next: use bool type for idx 0 or 1
  template <typename T>
  T gen_random(bool next);

  template <typename T, template <typename> class Tensor>
  void gen_random(Tensor<T> &tensor, bool next);

  template <typename T> T gen_random_private();

  template <typename T, template <typename> class Tensor>
  void gen_random_private(Tensor<T> &tensor);

  template <typename T>
  T gen_zero_sharing_arithmetic();

  template <typename T, template <typename> class Tensor>
  void gen_zero_sharing_arithmetic(Tensor<T> &tensor);

  template <typename T>
  T gen_zero_sharing_boolean();

  template <typename T, template <typename> class Tensor>
  void gen_zero_sharing_boolean(Tensor<T> &tensor);

#ifdef USE_CUDA
  static cudaStream_t _s_stream;
#endif

protected:
  virtual PseudorandomNumberGenerator& get_prng(size_t idx) = 0;

private:
  size_t _num_party;
  size_t _party;
  std::shared_ptr<AbstractNetwork> _network;
};

} // namespace mpc
} //namespace paddle

#include "./abstract_context_impl.h"
