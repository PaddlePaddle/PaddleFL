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
  AbstractContext() = default;
  AbstractContext(const AbstractContext &other) = delete;

  AbstractContext &operator=(const AbstractContext &other) = delete;

  virtual void init(size_t party, std::shared_ptr<AbstractNetwork> network, block seed,
            block seed2) = 0;

  void set_party(size_t party) {
    PADDLE_ENFORCE_LT(party, _num_party,
                     "party idx should less than %d.",
                     _num_party);
    _party = party;
  }

  void set_num_party(size_t num_party) {
    PADDLE_ENFORCE_EQ(num_party == 2 || num_party == 3, true,
                     "2 or 3 party protocol is supported.");
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
    _prng[idx].set_seed(seed);
  }

  size_t party() const { return _party; }

  size_t pre_party() const { return (_party + _num_party - 1) % _num_party; }

  size_t next_party() const { return (_party + 1) % _num_party; }

  template <typename T> T gen_random(bool next) { return _prng[next].get<T>(); }

  template <typename T, template <typename> class Tensor>
  void gen_random(Tensor<T> &tensor, bool next) {
    PADDLE_ENFORCE_EQ(_num_party, 3,
                     "`gen_random` API is for 3 party protocol.");
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this, next](T &val) { val = this->template gen_random<T>(next); });
  }

  template <typename T> T gen_random_private() { return _prng[2].get<T>(); }

  template <typename T, template <typename> class Tensor>
  void gen_random_private(Tensor<T> &tensor) {
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this](T &val) { val = this->template gen_random_private<T>(); });
  }

  template <typename T> T gen_zero_sharing_arithmetic() {
    PADDLE_ENFORCE_EQ(_num_party, 3,
                     "`gen_zero_sharing_arithmetic` API is for 3 party protocol.");
    return _prng[0].get<T>() - _prng[1].get<T>();
  }

  template <typename T, template <typename> class Tensor>
  void gen_zero_sharing_arithmetic(Tensor<T> &tensor) {
    PADDLE_ENFORCE_EQ(_num_party, 3,
                     "`gen_zero_sharing_arithmetic` API is for 3 party protocol.");
    std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                  [this](T &val) {
                    val = this->template gen_zero_sharing_arithmetic<T>();
                  });
  }

  template <typename T> T gen_zero_sharing_boolean() {
    PADDLE_ENFORCE_EQ(_num_party, 3,
                     "`gen_zero_sharing_boolean` API is for 3 party protocol.");
    return _prng[0].get<T>() ^ _prng[1].get<T>();
  }

  template <typename T, template <typename> class Tensor>
  void gen_zero_sharing_boolean(Tensor<T> &tensor) {
    PADDLE_ENFORCE_EQ(_num_party, 3,
                     "`gen_zero_sharing_boolean` API is for 3 party protocol.");
    std::for_each(
        tensor.data(), tensor.data() + tensor.numel(),
        [this](T &val) { val = this->template gen_zero_sharing_boolean<T>(); });
  }

  template <typename T, template <typename> class Tensor>
  void ot(size_t sender, size_t receiver, size_t helper,
          const Tensor<T> *choice, const Tensor<T> *m[2], Tensor<T> *buffer[2],
          Tensor<T> *ret) {
    // TODO: check tensor shape equals
    PADDLE_ENFORCE_EQ(_num_party, 3,
                     "`ot` API is for 3 party protocol.");
    const size_t numel = buffer[0]->numel();
    if (party() == sender) {
      bool common = helper == next_party();
      this->template gen_random(*buffer[0], common);
      this->template gen_random(*buffer[1], common);
      for (size_t i = 0; i < numel; ++i) {
        buffer[0]->data()[i] ^= m[0]->data()[i];
        buffer[1]->data()[i] ^= m[1]->data()[i];
      }
      network()->template send(receiver, *buffer[0]);
      network()->template send(receiver, *buffer[1]);

    } else if (party() == helper) {
      bool common = sender == next_party();

      this->template gen_random(*buffer[0], common);
      this->template gen_random(*buffer[1], common);

      for (size_t i = 0; i < numel; ++i) {
        // TODO: check if choice is one bit
        buffer[0]->data()[i] =
            choice->data()[i] ? buffer[1]->data()[i] : buffer[0]->data()[i];
      }
      network()->template send(receiver, *buffer[0]);
    } else if (party() == receiver) {
      network()->template recv(sender, *buffer[0]);
      network()->template recv(sender, *buffer[1]);
      network()->template recv(helper, *ret);
      size_t i = 0;
      std::for_each(ret->data(), ret->data() + numel,
                    [&buffer, &i, choice, ret](T &in) {
                      // TODO: check if choice is one bit
                      bool c = choice->data()[i];
                      in ^= buffer[c]->data()[i];
                      ++i;
                    });
    }
  }

private:
  size_t _num_party;
  size_t _party;
  std::shared_ptr<AbstractNetwork> _network;
  PseudorandomNumberGenerator _prng[3];

};

} // namespace mpc

} //namespace paddle
