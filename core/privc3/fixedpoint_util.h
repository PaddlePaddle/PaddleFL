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

#include <cmath>
#include <iostream>

#include "prng_utils.h"

namespace aby3 {
template <typename T, size_t N> class FixedPointUtil {
public:
  static double reveal(T *shares[3]) {
    // reveal
    T sum = *shares[0] + *shares[1] + *shares[2];
    // to double
    int neg = sum < 0 ? -1 : 1;
    sum = sum * neg;
    T high = sum >> N;
    T low = sum & (((T)1 << N) - 1);
    double ret = high + low / pow(2, N);
    return neg * ret;
  }

  static void share(double input, T *ret[3]) {
    // to int
    int neg = input < 0 ? -1 : 1;
    double val = input * neg;
    T high = val;
    double low = val - high;
    T ll_in = ((T)high << N) + (T)(low * pow(2, N));
    ll_in *= neg;
    // share
    *ret[0] = _s_prng.get<T>();
    *ret[1] = _s_prng.get<T>();
    *ret[2] = ll_in - *ret[0] - *ret[1];
  }

  static PseudorandomNumberGenerator _s_prng;
};

template <typename T, size_t N>
PseudorandomNumberGenerator
    FixedPointUtil<T, N>::_s_prng(block_from_dev_urandom());

} // namespace aby3
