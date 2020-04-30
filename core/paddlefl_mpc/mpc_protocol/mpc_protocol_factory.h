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

// Description:
// a factory class, help give pre-defined mpc protocol instances with given name

#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "mpc_protocol.h"

namespace paddle {
namespace mpc {

class MpcProtocolFactory {
public:
  using Creator = std::function<std::shared_ptr<MpcProtocol>()>;
  using CreatorMap = std::unordered_map<std::string, Creator>;

  MpcProtocolFactory() = delete;

  static void register_protocol();

  static std::shared_ptr<MpcProtocol> build(const std::string &name);

private:
  static bool _is_initialized;
  static CreatorMap _creator_map;

  static inline std::string to_lowercase(const std::string &str) {
    std::string orig_str(str);
    std::transform(orig_str.begin(), orig_str.end(), orig_str.begin(),
                   ::tolower);
    return orig_str;
  }
};

} // mpc
} // paddle
