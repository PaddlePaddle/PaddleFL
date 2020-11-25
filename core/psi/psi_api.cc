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

#include "psi_api.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>

#include "net_io.h"
#include "psi.h"
#include "../common/rand_utils.h"

namespace psi {

class PsiApi {
public:
  PsiApi() = default;

  ~PsiApi(){};

  static void set_psi_timeout(int timeout_s) { _s_timeout_s = timeout_s; }

  void psi_send(const std::set<std::string> &in,
                std::atomic<int> *psi_progress) {

    std::atomic<int> psi_prog(0);

    if (!psi_progress) {
      psi_progress = &psi_prog;
    }

    *psi_progress = 0;

    size_t local_size = in.size();
    size_t remote_size = 0;

    _io->send_data(&local_size, sizeof(size_t));

    _io->recv_data_with_timeout(&remote_size, sizeof(size_t));

    if (local_size == 0 || remote_size == 0) {
      *psi_progress = 100;
      return;
    }

    auto random_seed = common::block_from_dev_urandom();

    std::unique_ptr<PsiSender> sender;
    {
      std::lock_guard<std::mutex> guard(_s_init_mutex);
      sender = std::unique_ptr<PsiSender>(
          new PsiSender(local_size, remote_size, random_seed));
    }

    // 512 for ot size
    std::array<std::array<std::array<uint8_t, common::g_point_buffer_len>, 2>, 512>
        recv_input;

    _io->recv_data_with_timeout(&recv_input, sizeof(recv_input));

    std::array<std::array<uint8_t, common::g_point_buffer_len>, 512> send_back;
    for (size_t i = 0; i < 512; ++i) {
      send_back[i] = sender->np_ot().recv(i, recv_input[i]);
    }

    _io->send_data(&send_back, sizeof(send_back));

    *psi_progress = 2;

    sender->init_offline(in);

    *psi_progress = 18;

    sender->sync();

    *psi_progress = 30;

    size_t cuckoo_size = 0;

    _io->recv_data_with_timeout(&cuckoo_size, sizeof(size_t));

    double recv_times = cuckoo_size / (_s_recv_step_len / sizeof(Block512));

    double prog_ = 30;

    for (size_t offset = 0; offset < cuckoo_size;) {

      auto recv_len =
          std::min(_s_recv_step_len / sizeof(Block512), cuckoo_size - offset);

      std::vector<Block512> masks(recv_len);

      _io->recv_data_with_timeout(masks.data(), recv_len * sizeof(Block512));

      sender->recv_masks(offset, offset + recv_len, masks);

      prog_ += 45.0 / recv_times;

      *psi_progress = prog_;

      offset += recv_len;
    }

    *psi_progress = 75;

    for (size_t idx = 0; idx < 3; ++idx) {
      const auto &vec = sender->send_oprf_outputs(idx);

      const uint8_t *data = vec.data();

      size_t len = vec.size();

      _io->send_data(&len, sizeof(len));
      _io->send_data(data, len);

      *psi_progress += 7;
    }

    size_t stash_size = 0;

    _io->recv_data_with_timeout(&stash_size, sizeof(size_t));

    for (size_t i = 0; i < stash_size; ++i) {
      auto bin_idx = cuckoo_size + i;
      size_t hash_idx = 3 + i;

      std::vector<Block512> masks(1);
      _io->recv_data_with_timeout(masks.data(),
                                  masks.size() * sizeof(Block512));

      sender->recv_masks(bin_idx, bin_idx + 1, masks);

      const auto &vec = sender->send_oprf_outputs(hash_idx);

      size_t len = vec.size();
      _io->send_data(&len, sizeof(size_t));
      _io->send_data(vec.data(), vec.size());
    }

    *psi_progress = 100;
    return;
  }

  int psi_recv(const std::set<std::string> &in, std::vector<std::string> *out,
               std::atomic<int> *psi_progress) {

    std::atomic<int> psi_prog(0);

    if (out) {
      out->clear();
    }

    if (!psi_progress) {
      psi_progress = &psi_prog;
    }

    *psi_progress = 0;

    size_t local_size = in.size();
    size_t remote_size = 0;

    _io->recv_data_with_timeout(&remote_size, sizeof(size_t));

    _io->send_data(&local_size, sizeof(size_t));

    if (local_size == 0 || remote_size == 0) {
      *psi_progress = 100;
      return 0;
    }

    auto random_seed = common::block_from_dev_urandom();

    std::unique_ptr<PsiReceiver> recver;
    {
      std::lock_guard<std::mutex> guard(_s_init_mutex);
      recver = std::unique_ptr<PsiReceiver>(
          new PsiReceiver(remote_size, local_size, random_seed));
    }

    // ot size = 512
    std::array<std::array<std::array<uint8_t, common::g_point_buffer_len>, 2>, 512>
        to_send;

    for (size_t i = 0; i < 512; ++i) {
      to_send[i] = recver->np_ot().send_pre(i);
    }
    _io->send_data(&to_send, sizeof(to_send));
    std::array<std::array<uint8_t, common::g_point_buffer_len>, 512> recved;
    _io->recv_data_with_timeout(&recved, sizeof(recved));
    for (size_t i = 0; i < 512; ++i) {
      recver->np_ot().send_post(i, recved[i]);
    }

    *psi_progress = 2;

    recver->init_offline(in);

    *psi_progress = 18;

    recver->sync();

    *psi_progress = 30;

    size_t cuckoo_size = recver->cuckoo_bins_num();
    size_t stash_size = recver->stash_bins_num();

    _io->send_data(&cuckoo_size, sizeof(size_t));

    auto masks = recver->send_masks(0, cuckoo_size);

    *psi_progress = 75;

    _io->send_data(masks.data(), masks.size() * sizeof(Block512));

    const auto oprf_len = recver->oprf_output_len();

    double prog_ = 75;

    for (size_t idx = 0; idx < 3; ++idx) {
      size_t len = 0;
      _io->recv_data_with_timeout(&len, sizeof(len));
      std::vector<char> buf(oprf_len);

      double recv_times = len * 1.0 / _s_recv_step_len;
      for (size_t offset = 0; offset < len;) {
        std::vector<std::string> output_buf;
        size_t round_len = std::min(_s_recv_step_len, size_t(len - offset));
        size_t round_end = offset + round_len;
        for (; offset < round_end; offset += oprf_len) {
          _io->recv_data_with_timeout(buf.data(), oprf_len);
          output_buf.emplace_back(std::string(buf.data(), oprf_len));
        }
        recver->recv_oprf_outputs(idx, output_buf);

        prog_ += 7.0 / recv_times;
        *psi_progress = prog_;
      }
    }

    *psi_progress = 96;

    _io->send_data(&stash_size, sizeof(size_t));

    for (size_t i = 0; i < stash_size; ++i) {
      auto bin_idx = cuckoo_size + i;
      size_t hash_idx = 3 + i;
      auto masks = recver->send_masks(bin_idx, bin_idx + 1);

      _io->send_data(masks.data(), masks.size() * sizeof(Block512));

      size_t len = 0;
      _io->recv_data_with_timeout(&len, sizeof(len));
      std::vector<std::string> output_buf;
      for (size_t offset = 0; offset < len; offset += oprf_len) {
        std::vector<char> buf(oprf_len);
        _io->recv_data_with_timeout(buf.data(), oprf_len);
        output_buf.emplace_back(std::string(buf.data(), oprf_len));
      }
      recver->recv_oprf_outputs(hash_idx, output_buf);
    }
    *psi_progress = 99;

    if (out) {
      *out = recver->output();
    }

    *psi_progress = 100;

    return 0;
  }

  NetIO *_io;

  static int _s_timeout_s;

private:
  static std::mutex _s_init_mutex;

  static const size_t _s_recv_step_len;
};

std::mutex PsiApi::_s_init_mutex;
const size_t PsiApi::_s_recv_step_len = 0x1000000;

// default sync sock, no timeout
int PsiApi::_s_timeout_s = 0;

int psi_send(int port, const std::set<std::string> &in,
             std::atomic<int> *psi_progress) {
  try {
    PsiApi sender;

    NetIO io(nullptr, port, true, PsiApi::_s_timeout_s);

    sender._io = &io;

    sender.psi_send(in, psi_progress);

  } catch (const std::exception &e) {
    if (psi_progress) {
      *psi_progress = -1;
    }
    throw;
  }
  return 0;
}

int psi_recv(const std::string &remote_ip, int port,
             const std::set<std::string> &in, std::vector<std::string> *out,
             std::atomic<int> *psi_progress) {
  try {
    PsiApi recver;

    NetIO io(remote_ip.c_str(), port, true, PsiApi::_s_timeout_s);

    recver._io = &io;

    recver.psi_recv(in, out, psi_progress);

  } catch (const std::exception &e) {
    if (psi_progress) {
      *psi_progress = -1;
    }
    throw;
  }
  return 0;
}

void set_psi_timeout(int timeout_s) { PsiApi::set_psi_timeout(timeout_s); }
} // namespace psi
