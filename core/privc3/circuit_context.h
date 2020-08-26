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
#include <algorithm>

#include "core/paddlefl_mpc/mpc_protocol/abstract_network.h"
#include "prng_utils.h"

namespace aby3 {

using AbstractNetwork = paddle::mpc::AbstractNetwork;

class CircuitContext {
public:
    CircuitContext(size_t party,
                   std::shared_ptr<AbstractNetwork> network,
                   const block& seed = g_zero_block,
                   const block& seed2 = g_zero_block) {
        init(party, network, seed, seed2);
    }

    CircuitContext(const CircuitContext& other) = delete;

    CircuitContext& operator=(const CircuitContext& other) = delete;

    void init(size_t party,
              std::shared_ptr<AbstractNetwork> network,
              block seed,
              block seed2) {
        set_party(party);
        set_network(network);

        if (equals(seed, g_zero_block)) {
            seed = block_from_dev_urandom();
        }

        if (equals(seed2, g_zero_block)) {
            seed2 = block_from_dev_urandom();
        }
        set_random_seed(seed, 0);
        // seed2 is private
        set_random_seed(seed2, 2);

        // 3 for 3-party computation
        size_t party_pre = (this->party() - 1 + 3) % 3;
        size_t party_next = (this->party() + 1) % 3;

        if (party == 1) {
            block recv_seed = this->network()->template recv<block>(party_next);
            this->network()->template send(party_pre, seed);
            seed = recv_seed;
        } else {
            this->network()->template send(party_pre, seed);
            seed = this->network()->template recv<block>(party_next);
        }

        set_random_seed(seed, 1);
    }

    void set_party(size_t party) {
        if (party >= 3) {
            // exception handling
        }
        _party = party;
    }

    void set_network(std::shared_ptr<AbstractNetwork> network) {
        _network = network;
    }
<<<<<<< HEAD

    AbstractNetwork* network() {
        return _network.get();
    }

    void set_random_seed(const block& seed, size_t idx) {
        if (idx >= 3) {
            // exception handling
        }
        _prng[idx].set_seed(seed);
    }

    size_t party() const {
        return _party;
    }

    size_t pre_party() const {
        return (_party + 3 - 1) % 3;
    }

    size_t next_party() const {
        return (_party + 1) % 3;
    }

    template <typename T>
    T gen_random(bool next) {
        return _prng[next].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_random(Tensor<T>& tensor, bool next) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this, next](T& val) {
                          val = this->template gen_random<T>(next);
                      });
    }

    template <typename T>
    T gen_random_private() {
        return _prng[2].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_random_private(Tensor<T>& tensor) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this](T& val) {
                          val = this->template gen_random_private<T>();
                      });
    }

    template <typename T>
    T gen_zero_sharing_arithmetic() {
        return _prng[0].get<T>() - _prng[1].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_zero_sharing_arithmetic(Tensor<T>& tensor) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this](T& val) {
                          val = this->template gen_zero_sharing_arithmetic<T>();
                      });
    }

    template <typename T>
    T gen_zero_sharing_boolean() {
        return _prng[0].get<T>() ^ _prng[1].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_zero_sharing_boolean(Tensor<T>& tensor) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this](T& val) {
                          val = this->template gen_zero_sharing_boolean<T>();
                      });
    }

    template<typename T, template <typename> class Tensor>
=======

    AbstractNetwork* network() {
        return _network.get();
    }

    void set_random_seed(const block& seed, size_t idx) {
        if (idx >= 3) {
            // exception handling
        }
        _prng[idx].set_seed(seed);
    }

    size_t party() const {
        return _party;
    }

    size_t pre_party() const {
        return (_party + 3 - 1) % 3;
    }

    size_t next_party() const {
        return (_party + 1) % 3;
    }

    template <typename T>
    T gen_random(bool next) {
        return _prng[next].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_random(Tensor<T>& tensor, bool next) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this, next](T& val) {
                          val = this->template gen_random<T>(next);
                      });
    }

    template <typename T>
    T gen_random_private() {
        return _prng[2].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_random_private(Tensor<T>& tensor) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this](T& val) {
                          val = this->template gen_random_private<T>();
                      });
    }

    template <typename T>
    T gen_zero_sharing_arithmetic() {
        return _prng[0].get<T>() - _prng[1].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_zero_sharing_arithmetic(Tensor<T>& tensor) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this](T& val) {
                          val = this->template gen_zero_sharing_arithmetic<T>();
                      });
    }

    template <typename T>
    T gen_zero_sharing_boolean() {
        return _prng[0].get<T>() ^ _prng[1].get<T>();
    }

    template<typename T, template <typename> class Tensor>
    void gen_zero_sharing_boolean(Tensor<T>& tensor) {
        std::for_each(tensor.data(), tensor.data() + tensor.numel(),
                      [this](T& val) {
                          val = this->template gen_zero_sharing_boolean<T>();
                      });
    }

    template<typename T, template <typename> class Tensor>
>>>>>>> 5a09665c36ffb7eae2288b3f837d3be18091c259
    void ot(size_t sender, size_t receiver, size_t helper,
            const Tensor<T>* choice, const Tensor<T>* m[2],
            Tensor<T>* buffer[2], Tensor<T>* ret) {
        // TODO: check tensor shape equals
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
                buffer[0]->data()[i] = choice->data()[i] & 1 ?
                    buffer[1]->data()[i] : buffer[0]->data()[i];
            }
            network()->template send(receiver, *buffer[0]);
        } else if (party() == receiver) {
            network()->template recv(sender, *buffer[0]);
            network()->template recv(sender, *buffer[1]);
            network()->template recv(helper, *ret);
            size_t i = 0;
            std::for_each(ret->data(), ret->data() + numel, [&buffer, &i, choice, ret](T& in) {
                          bool c = choice->data()[i] & 1;
                          in ^= buffer[c]->data()[i];
                          ++i;}
                          );
        }
    }

private:
    size_t _party;

    std::shared_ptr<AbstractNetwork> _network;

    PseudorandomNumberGenerator _prng[3];

};

} // namespace aby3
