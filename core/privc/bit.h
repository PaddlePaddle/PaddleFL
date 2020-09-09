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

#include <vector>
#include <limits>

#include "core/privc/privc_context.h"
#include "core/privc/crypto.h"
#include "core/privc/triplet_generator.h"
#include "core/privc/common.h"
#include "core/privc/ot.h"

namespace privc {

block garbled_and(block a, block b);

block garbled_share(bool val);

std::vector<block> garbled_share(int64_t val);

std::vector<block> garbled_share(const std::vector<int64_t>& val);

class Bit {
public:
    block _share;

public:
    Bit() : _share(psi::ZeroBlock) {}

    Bit(bool val, size_t party_in) {
        if (party_in == 0) {
            if (party() == 0) {
                _share = privc_ctx()->gen_random_private<block>();
                block to_send = _share;
                if (val) {
                    to_send ^= ot()->garbled_delta();
                }
                net()->send(next_party(), to_send);

            } else {
                _share = net()->recv<block>(next_party());
            }
        } else {
            _share = garbled_share(val);
        }
    }

    ~Bit() {}

    Bit operator^(const Bit &rhs) const {
        Bit ret;
        ret._share = _share ^ rhs._share;
        return ret;
    }

    block& share() {
      return _share;
    }

    const block& share() const {
      return _share;
    }

    Bit operator&(const Bit &rhs) const {
        Bit ret;
        ret._share = garbled_and(_share, rhs._share);
        return ret;
    }

    Bit operator|(const Bit &rhs) const { return *this ^ rhs ^ (*this & rhs); }

    Bit operator~() const {
        Bit ret;

        ret._share = _share;

        if (party() == 0) {
            ret._share ^= ot()->garbled_delta();
        }

        return ret;
    }

    Bit operator&&(const Bit &rhs) const {
        return *this & rhs;
    }

    Bit operator||(const Bit &rhs) const {
        return *this | rhs;
    }

    Bit operator!() const {
        return ~*this;
    }
    u8 lsb() const {
      u8 ret = block_lsb(_share);
      return ret & (u8)1;
    }

    bool reconstruct() const {
        u8 remote;
        u8 local = block_lsb(_share);

        if (party() == 0) {
            remote = net()->recv<u8>(next_party());
            net()->send(next_party(), local);
        } else {
            net()->send(next_party(), local);
            remote = net()->recv<u8>(next_party());
        }

        return remote ^ local;
    }

};

std::vector<bool> reconstruct(std::vector<Bit> bits,
              size_t party_in);

using Bool = Bit;

} // namespace privc

