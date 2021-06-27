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
// limitations under the License.nguage governing permissions and

#include "mpc_protocol_factory.h"
#include "aby3_protocol.h"
#include "privc_protocol.h"

namespace paddle {
namespace mpc {

void MpcProtocolFactory::register_protocol() {
  if (!_is_initialized) {
    _creator_map.insert({"aby3", std::make_shared<Aby3Protocol>});
    _creator_map.insert({"privc", std::make_shared<PrivCProtocol>});
  }
  _is_initialized = true;
}

std::shared_ptr<MpcProtocol>
MpcProtocolFactory::build(const std::string &name) {
  if (!_is_initialized) {
    register_protocol();
  }
  auto where = _creator_map.find(to_lowercase(name));
  if (where == _creator_map.end()) {
    return nullptr;
  }
  return where->second();
}

MpcProtocolFactory::CreatorMap MpcProtocolFactory::_creator_map;
bool MpcProtocolFactory::_is_initialized = false;

} // mpc
} // paddle
