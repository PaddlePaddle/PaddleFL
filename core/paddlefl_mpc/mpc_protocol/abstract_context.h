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
#include "core/privc3/prng_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

namespace mpc {

using block = psi::block;
using PseudorandomNumberGenerator = psi::PseudorandomNumberGenerator;

class AbstractContext {
public:
  AbstractContext(size_t party, std::shared_ptr<AbstractNetwork> network) {
    init(party, network);
  };
  AbstractContext(const AbstractContext &other) = delete;

  AbstractContext &operator=(const AbstractContext &other) = delete;

  void init(size_t party, std::shared_ptr<AbstractNetwork> network) {
    set_party(party);
    set_network(network);
  }

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

  void set_random_seed(const block &seed, size_t idx) {
    PADDLE_ENFORCE_LE(idx, _num_party,
                     "prng idx should be less and equal to %d.",
                     _num_party);
    get_prng(idx).set_seed(seed);
  }

  size_t party() const { return _party; }

  size_t pre_party() const {
    return (_party + _num_party - 1) % _num_party;
  }

  size_t next_party() const {
    return (_party + 1) % _num_party;
  }

  // generate random from prng[0] or prng[1]
  // @param next: use bool type for idx 0 or 1
  template <typename T> T gen_random(bool next) {
    return get_prng(next).get<T>();
  }

  template <typename T, template <typename> class Tensor>
  void gen_random(Tensor<T> &tensor, bool next) {
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this, next](T &val) { val = this->template gen_random<T>(next); });
  }

  template <typename T> T gen_random_private() { return get_prng(2).get<T>(); }

  template <typename T, template <typename> class Tensor>
  void gen_random_private(Tensor<T> &tensor) {
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this](T &val) { val = this->template gen_random_private<T>(); });
  }

  template <typename T> T gen_zero_sharing_arithmetic() {
    return get_prng(0).get<T>() - get_prng(1).get<T>();
  }

  template <typename T, template <typename> class Tensor>
  void gen_zero_sharing_arithmetic(Tensor<T> &tensor) {
    std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                  [this](T &val) {
                    val = this->template gen_zero_sharing_arithmetic<T>();
                  });
  }

  template <typename T> T gen_zero_sharing_boolean() {
    return get_prng(0).get<T>() ^ get_prng(1).get<T>();
  }

  template <typename T, template <typename> class Tensor>
  void gen_zero_sharing_boolean(Tensor<T> &tensor) {
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this](T &val) { val = this->template gen_zero_sharing_boolean<T>(); });
  }

protected:
  virtual PseudorandomNumberGenerator& get_prng(size_t idx) = 0;

private:
  size_t _num_party;
  size_t _party;
  std::shared_ptr<AbstractNetwork> _network;
  PseudorandomNumberGenerator _prng[3];

};

} // namespace mpc

} //namespace paddle
