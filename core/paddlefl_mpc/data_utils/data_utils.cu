/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <atomic>
#include <set>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/privc3/fixedpoint_util.h"
#include "core/privc/fixedpoint_util.h"
#include "core/paddlefl_mpc/mpc_protocol/aby3_operators.h"
#include "core/psi/psi_api.h"

namespace py = pybind11;

namespace mpc {

// split plaintext into three shares.
template<typename T, size_t N>
py::array_t<T> aby3_share(double input) {
    size_t share_num = 3;
    auto shares = py::array_t<T>(share_num);
    py::buffer_info shares_buf = shares.request();
    T* shares_buf_ptr = (T*)shares_buf.ptr;
    T* ret_ptr[share_num];
    for (size_t i = 0; i < share_num; ++i) {
        ret_ptr[i] = &shares_buf_ptr[i];
    }

    aby3::FixedPointUtils<T, N>::share(input, ret_ptr);

    return shares;
}

// combine three shares to reveal plaintext.
template<typename T, size_t N>
double aby3_reveal(py::array_t<T> shares) {
    size_t share_num = 3;
    py::buffer_info shares_buf = shares.request();
    T *shares_buf_ptr = (T *) shares_buf.ptr;
    T *ret[share_num];

    for (size_t idx = 0; idx < share_num; ++idx) {
        ret[idx] = &shares_buf_ptr[idx];
    }

    double result = aby3::FixedPointUtils<T, N>::reveal(ret);

    return result;
}

// split plaintext into two shares.
template<typename T, size_t N>
py::array_t<T> privc_share(double input) {
    size_t share_num = 2;
    auto shares = py::array_t<T>(share_num);
    py::buffer_info shares_buf = shares.request();
    T* shares_buf_ptr = (T*)shares_buf.ptr;
    T* ret_ptr[share_num];
    for (size_t i = 0; i < share_num; ++i) {
        ret_ptr[i] = &shares_buf_ptr[i];
    }

    privc::FixedPointUtils<T, N>::share(input, ret_ptr);

    return shares;
}

// combine two shares to reveal plaintext.
template<typename T, size_t N>
double privc_reveal(py::array_t<T> shares) {
    size_t share_num = 2;
    py::buffer_info shares_buf = shares.request();
    T *shares_buf_ptr = (T *) shares_buf.ptr;
    T *ret[share_num];

    for (size_t idx = 0; idx < share_num; ++idx) {
        ret[idx] = &shares_buf_ptr[idx];
    }

    double result = privc::FixedPointUtils<T, N>::reveal(ret);

    return result;
}

// call psi_send
int send_psi(int port, const std::set<std::string>& input) {
    std::atomic<int> prog(0);
    return psi::psi_send(port, input, &prog);
}

// call psi_recv
std::vector<std::string> recv_psi(const std::string &remote_ip,
                                  int port,
                                  const std::set<std::string>& input) {
    std::vector<std::string> output;
    std::atomic<int> prog(0);
    int ret = psi::psi_recv(remote_ip, port, input, &output, &prog);
    if (ret != 0) {
        output.clear();
        return output;
    }
    return output;
}

PYBIND11_MODULE(mpc_data_utils, m)
{
    // optional module docstring
    m.doc() = "pybind11 paddle-mpc plugin: data_utils (share, reveal, psi)";

    m.def("aby3_share", &aby3_share<long long, paddle::mpc::ABY3_SCALING_FACTOR>,
          "split plaintext into three shares.");
    m.def("aby3_reveal", &aby3_reveal<long long, paddle::mpc::ABY3_SCALING_FACTOR>,
          "combine three shares to reveal plaintext.");

    m.def("privc_share", &privc_share<long long, privc::PRIVC_FIXED_POINT_SCALING_FACTOR>,
          "split plaintext into two shares.");
    m.def("privc_reveal", &privc_reveal<long long, privc::PRIVC_FIXED_POINT_SCALING_FACTOR>,
          "combine two shares to reveal plaintext.");

    m.def("send_psi", &send_psi, "Send input in two party PSI.");
    m.def("recv_psi", &recv_psi, "Send input and return PSI result as output in two party PSI.");

    int64_t ONE = 1;
    m.attr("mpc_one_share") = (ONE << paddle::mpc::ABY3_SCALING_FACTOR) / 3; // todo: remove
    m.attr("aby3_one_share") = (ONE << paddle::mpc::ABY3_SCALING_FACTOR) / 3;
    m.attr("privc_one_share") = (ONE <<  privc::PRIVC_FIXED_POINT_SCALING_FACTOR) / 2;
}

}  // namespace mpc


