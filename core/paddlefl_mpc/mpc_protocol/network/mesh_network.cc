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

#include "mesh_network.h"

#include "gloo/common/string.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/device.h"
#include "gloo/transport/tcp/device.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace mpc {

// for test purpose
void MeshNetwork::init() {
  if (_is_initialized) {
    return;
  }

  auto context = 
      std::make_shared<gloo::rendezvous::Context>(_party_id, _net_size);
  auto dev = gloo::transport::tcp::CreateDevice(_local_addr.c_str());
  auto prefix_store = gloo::rendezvous::PrefixStore(_store_prefix, *_store);

  context->connectFullMesh(prefix_store, dev);

  _rendezvous_ctx = std::move(context);
  _is_initialized = true;
}

void MeshNetwork::send(size_t party, const void *data, size_t size) {
  PADDLE_ENFORCE_NOT_NULL(data);
  PADDLE_ENFORCE(_is_initialized);

  auto unbounded_buf =
      _rendezvous_ctx->createUnboundBuffer(const_cast<void *>(data), size);
  unbounded_buf->send(party, 0UL /*slot*/);
  unbounded_buf->waitSend();
}

void MeshNetwork::recv(size_t party, void *data, size_t size) {
  PADDLE_ENFORCE_NOT_NULL(data);
  PADDLE_ENFORCE(_is_initialized);

  auto unbounded_buf = _rendezvous_ctx->createUnboundBuffer(data, size);
  unbounded_buf->recv(party, 0UL /*slot*/);
  unbounded_buf->waitRecv();
}

} // mpc
} // paddle
