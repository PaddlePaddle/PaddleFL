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

#include "mpc_instance.h"

namespace paddle {
namespace mpc {

thread_local std::once_flag MpcInstance::_s_init_flag;
thread_local std::once_flag MpcInstance::_s_name_init_flag;
thread_local bool MpcInstance::_s_name_initialized = false;
thread_local std::shared_ptr<MpcInstance> MpcInstance::_s_mpc_instance(nullptr);
thread_local std::shared_ptr<MpcProtocol> MpcInstance::_s_mpc_protocol(nullptr);
thread_local std::string MpcInstance::_protocol_name = "";
} // namespace framework
} // namespace paddle
