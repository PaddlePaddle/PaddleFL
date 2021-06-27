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
// abstract mpc operation interface

#pragma once
#include <string>
#include <unordered_map>

namespace paddle {
namespace mpc {

class MpcConfig {
public:
  MpcConfig() {}

  MpcConfig(const MpcConfig &config) = default;

  int get_int(const std::string &key, int default_val = 0) const {
    auto got = _prop_map.find(key);
    if (got != _prop_map.end()) {
      auto ret = got->second;
      return std::stoi(ret);
    }
    return default_val;
  }

  // get value accoding to specified key, an empty string is returned otherwise
  std::string get(const std::string &key,
                  const std::string &default_val = std::string()) const {
    auto got = _prop_map.find(key);
    if (got != _prop_map.end()) {
      return got->second;
    }
    return default_val;
  }

  // set the config item. if an item with same key exists, it will be
  // overwritten.
  MpcConfig &set(const std::string &key, const std::string &value) {
    _prop_map[key] = value;
    return *this;
  }

  MpcConfig &set_int(const std::string &key, const int value) {
    return set(key, std::to_string(value));
  }

private:
  std::unordered_map<std::string, std::string> _prop_map;
  
public:
  static const std::string ROLE;
  static const std::string NET_SIZE;
  static const std::string LOCAL_ADDR;
  static const std::string NET_SERVER_ADDR;
  static const std::string NET_SERVER_PORT;
  static const std::string ENDPOINTS;
  static const std::string NETWORK_MODE;

  // default values
  static const std::string LOCAL_ADDR_DEFAULT;
  static const std::string NET_SERVER_ADDR_DEFAULT;
  static const int NET_SERVER_PORT_DEFAULT;
  static const std::string ENDPOINTS_DEFAULT;
  static const std::string NETWORK_MODE_DEFAULT;
};

} // mpc
} // paddle
