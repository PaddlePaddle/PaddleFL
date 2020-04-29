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

#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "gtest/gtest.h"

#include "fixedpoint_tensor.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/paddlefl_mpc/mpc_protocol/mesh_network.h"

namespace aby3 {

using g_ctx_holder = paddle::mpc::ContextHolder;
using Fix64N16 = FixedPointTensor<int64_t, 16>;

class FixedTensorTest : public ::testing::Test {
public:
  paddle::platform::CPUDeviceContext _cpu_ctx;
  std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
  std::shared_ptr<CircuitContext> _mpc_ctx[3];
  std::shared_ptr<gloo::rendezvous::HashStore> _store;
  std::thread _t[3];
  std::shared_ptr<TensorAdapterFactory> _s_tensor_factory;

  void SetUp() {

    paddle::framework::OperatorBase *op = nullptr;
    paddle::framework::Scope scope;
    paddle::framework::RuntimeContext ctx({}, {});
    // only device_ctx is needed
    _exec_ctx = std::make_shared<paddle::framework::ExecutionContext>(
        *op, scope, _cpu_ctx, ctx, nullptr);

    _store = std::make_shared<gloo::rendezvous::HashStore>();

    std::thread t[3];
    for (size_t i = 0; i < 3; ++i) {
      _t[i] = std::thread(&FixedTensorTest::gen_mpc_ctx, this, i);
    }
    for (auto &ti : _t) {
      ti.join();
    }
    _s_tensor_factory = std::make_shared<PaddleTensorFactory>(&_cpu_ctx);
  }
  std::shared_ptr<paddle::mpc::MeshNetwork> gen_network(size_t idx) {
    return std::make_shared<paddle::mpc::MeshNetwork>(idx, "127.0.0.1", 3,
                                                      "test_prefix", _store);
  }
  void gen_mpc_ctx(size_t idx) {
    auto net = gen_network(idx);
    net->init();
    _mpc_ctx[idx] = std::make_shared<CircuitContext>(idx, net);
  }

  std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
    return _s_tensor_factory->template create<int64_t>(shape);
  }
};

std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
  return g_ctx_holder::tensor_factory()->template create<int64_t>(shape);
}

template <typename T, size_t N>
PaddleTensor<T>
test_fixedt_gen_paddle_tensor(std::vector<float> &input,
                              std::vector<size_t> &shape,
                              paddle::platform::CPUDeviceContext &cpu_ctx) {

  PaddleTensor<T> ret(&cpu_ctx);
  ret.reshape(shape);
  T *ret_ptr = ret.data();
  for (int i = 0; i < ret.numel(); i++) {
    *(ret_ptr + i) = (T)(input[i] * pow(2, N));
  }
  return ret;
}

template <typename T>
bool test_fixedt_check_tensor_eq(const TensorAdapter<T> *in1,
                                 const TensorAdapter<T> *in2,
                                 double precision = 0.0001) {
  // check shape
  std::vector<size_t> shape1, shape2;
  shape1 = in1->shape();
  shape2 = in2->shape();
  size_t scale = in1->scaling_factor();
  if (shape1.size() != shape2.size()) {
    std::cout << "shape size error: shape1.size: " << shape1.size()
              << "; shape2.size: " << shape2.size() << std::endl;
    return false;
  }
  for (int i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i]) {
      std::cout << "shape error!" << std::endl;
      return false;
    }
  }

  // check each element
  for (int i = 0; i < in1->numel(); i++) {
    if (std::abs(*(in1->data() + i) - *(in2->data() + i)) >
        precision * pow(2, scale)) {
      std::cout << "result error: inx: " << i
                << " in1[i] = " << *(in1->data() + i)
                << " in2[i] = " << *(in2->data() + i) << std::endl;
      return false;
    }
  }
  return true;
}

void test_fixedt_gen_shares(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> &out) {
  if (p == 0) {

    std::shared_ptr<TensorAdapter<int64_t>> out1_shared[3];
    std::shared_ptr<TensorAdapter<int64_t>> out2_shared[3];
    for (int i = 0; i < 3; i++) {
      out1_shared[i] =
          g_ctx_holder::tensor_factory()->create<int64_t>(out[0]->shape());
      out2_shared[i] =
          g_ctx_holder::tensor_factory()->create<int64_t>(out[0]->shape());
    }
    TensorAdapter<int64_t> *out1[3] = {
        out1_shared[0].get(), out1_shared[1].get(), out1_shared[2].get()};
    TensorAdapter<int64_t> *out2[3] = {
        out2_shared[0].get(), out2_shared[1].get(), out2_shared[2].get()};

    Fix64N16::share(in[0].get(), out1);
    Fix64N16::share(in[1].get(), out2);

    g_ctx_holder::mpc_ctx()->network()->template send(1, *out1[1]);
    g_ctx_holder::mpc_ctx()->network()->template send(1, *out1[2]);

    g_ctx_holder::mpc_ctx()->network()->template send(1, *out2[1]);
    g_ctx_holder::mpc_ctx()->network()->template send(1, *out2[2]);

    g_ctx_holder::mpc_ctx()->network()->template send(2, *out1[2]);
    g_ctx_holder::mpc_ctx()->network()->template send(2, *out1[0]);
    g_ctx_holder::mpc_ctx()->network()->template send(2, *out2[2]);
    g_ctx_holder::mpc_ctx()->network()->template send(2, *out2[0]);

    out1[0]->copy(out[0].get());
    out1[1]->copy(out[1].get());
    out2[0]->copy(out[2].get());
    out2[1]->copy(out[3].get());

  } else {
    std::shared_ptr<TensorAdapter<int64_t>> out3_shared[4];
    for (int i = 0; i < 4; i++) {
      out3_shared[i] =
          g_ctx_holder::tensor_factory()->create<int64_t>(out[0]->shape());
    }
    TensorAdapter<int64_t> *out3[4] = {
        out3_shared[0].get(), out3_shared[1].get(), out3_shared[2].get(),
        out3_shared[3].get()};
    for (int i = 0; i < 4; i++) {
      g_ctx_holder::mpc_ctx()->network()->template recv(0, *out3[i]);
      out3[i]->copy(out[i].get());
    }
  }
}

void test_fixedt_gen_shares(
    size_t p, std::shared_ptr<TensorAdapter<int64_t>> in,
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> &out) {

  if (p == 0) {

    std::shared_ptr<TensorAdapter<int64_t>> out1_shared[3];
    for (int i = 0; i < 3; i++) {
      out1_shared[i] =
          g_ctx_holder::tensor_factory()->create<int64_t>(out[0]->shape());
    }
    TensorAdapter<int64_t> *out1[3] = {
        out1_shared[0].get(), out1_shared[1].get(), out1_shared[2].get()};
    Fix64N16::share(in.get(), out1);

    g_ctx_holder::mpc_ctx()->network()->template send(1, *out1[1]);
    g_ctx_holder::mpc_ctx()->network()->template send(1, *out1[2]);

    g_ctx_holder::mpc_ctx()->network()->template send(2, *out1[2]);
    g_ctx_holder::mpc_ctx()->network()->template send(2, *out1[0]);

    out1[0]->copy(out[0].get());
    out1[1]->copy(out[1].get());

  } else {
    std::shared_ptr<TensorAdapter<int64_t>> out3_shared[2];
    for (int i = 0; i < 2; i++) {
      out3_shared[i] =
          g_ctx_holder::tensor_factory()->create<int64_t>(out[0]->shape());
    }
    TensorAdapter<int64_t> *out3[2] = {out3_shared[0].get(),
                                       out3_shared[1].get()};
    for (int i = 0; i < 2; i++) {
      g_ctx_holder::mpc_ctx()->network()->template recv(0, *out3[i]);
      out3[i]->copy(out[i].get());
    }
  }
}

void test_fixedt_share(size_t p, TensorAdapter<int64_t> *in,
                       TensorAdapter<int64_t> *ret) {

  if (in || ret) {
    TensorAdapter<int64_t> *output[3];
    for (int i = 0; i < 3; i++) {
      output[i] = new PaddleTensor<int64_t>(g_ctx_holder::device_ctx());
      dynamic_cast<PaddleTensor<int64_t> *>(output[i])->reshape(in->shape());
    }
    Fix64N16::share(in, output);
    output[0]->add(output[1], ret);
    ret->add(output[2], ret);

    for (int i = 0; i < 3; i++) {
      delete output[i];
    }
  }
}

void test_fixedt_add_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  Fix64N16 *result = new Fix64N16(temp[4].get(), temp[5].get());
  lhs->add(rhs, result);
  result->reveal(out);
}

void test_fixedt_add_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->add(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_sub_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  Fix64N16 *result = new Fix64N16(temp[4].get(), temp[5].get());
  lhs->sub(rhs, result);
  result->reveal(out);
}

void test_fixedt_sub_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->sub(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_neg_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->negative(result);
  result->reveal(out);
}

void test_fixedt_mul_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  Fix64N16 *result = new Fix64N16(temp[4].get(), temp[5].get());
  lhs->mul(rhs, result);
  result->reveal(out);
}

void test_fixedt_mul2_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  Fix64N16 *result = new Fix64N16(temp[4].get(), temp[5].get());
  lhs->mul2(rhs, result);
  result->reveal(out);
}

void test_fixedt_mul_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->mul(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_div_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->div(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_sum_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 2; i++) {
    temp.emplace_back(gen(in[0]->shape()));
  }

  for (int i = 2; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->sum(result);

  result->reveal(out);
}

void test_fixedt_poly_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  auto c_shape = out->shape();
  c_shape.insert(c_shape.begin(), 2);
  // construct coeff
  auto coeff = gen(c_shape);
  std::vector<int64_t> w;
  w.resize(2);
  w[0] = 1 << 16;
  w[1] = 1 << 16;

  auto c_ptr = coeff->data();
  for (size_t i = 0; i < w.size(); i++) {
    for (size_t j = 0; j < in[0]->numel(); j++) {
      *(c_ptr + i * in[0]->numel() + j) = w[i];
    }
  }
  coeff->scaling_factor() = 16;

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->polynomial(coeff.get(), result);
  result->reveal(out);
}

void test_fixedt_poly_wise_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());

  // constrct break_point
  auto shape = in[0]->shape();
  shape.insert(shape.begin(), 1);
  auto break_point = gen(shape);
  for (size_t i = 0; i < break_point->numel(); ++i) {
    break_point->data()[i] = 0;
  }
  break_point->scaling_factor() = 16;

  // contruct coeff
  std::vector<size_t> shape_ = {2, 2};
  auto in_shape = in[0]->shape();
  shape_.insert(shape_.end(), in_shape.begin(), in_shape.end());
  auto coeff = gen(shape_);
  int64_t *c_ptr = coeff->data();
  for (size_t i = 0; i < 4 * in[0]->numel(); i++) {
    *(c_ptr + i) = 1 << 16;
  }
  for (size_t i = in [0]->numel(); i < in[0]->numel() * 2; i++) {
    *(c_ptr + i) = 0;
  }
  coeff->scaling_factor() = 16;

  lhs->polynomial_piecewise(coeff.get(), break_point.get(), result);

  result->reveal(out);
}

void test_fixedt_relu_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->relu(result);
  result->reveal(out);
}

void test_fixedt_softmax_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->softmax(result);
  result->reveal(out);
}

void test_fixedt_sigmoid_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->sigmoid(result);
  result->reveal(out);
}

void test_fixedt_exp_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->exp(result);
  result->reveal(out);
}

void test_fixedt_mat_mul_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  Fix64N16 *result = new Fix64N16(temp[4].get(), temp[5].get());
  lhs->mat_mul(rhs, result);
  result->reveal(out);
}

void test_fixedt_mat_mul_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->mat_mul(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_dot_mul_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(in[0]->shape()));
  }
  for (int i = 4; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  Fix64N16 *result = new Fix64N16(temp[4].get(), temp[5].get());
  lhs->dot_mul(rhs, result);
  result->reveal(out);
}

void test_fixedt_dot_mul_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 2; i++) {
    temp.emplace_back(gen(in[0]->shape()));
  }
  for (int i = 2; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *result = new Fix64N16(temp[2].get(), temp[3].get());
  lhs->dot_mul(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_gt_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());

  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
  lhs->gt(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_gt_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
  lhs->gt(rhs, result);
  result->reveal(out);
}

void test_fixedt_lt_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());

  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
  lhs->lt(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_lt_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
  lhs->lt(rhs, result);
  result->reveal(out);
}

void test_fixedt_leq_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());

  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
  lhs->leq(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_leq_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
  lhs->leq(rhs, result);
  result->reveal(out);
}

void test_fixedt_geq_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());

  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
  lhs->geq(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_geq_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
  lhs->geq(rhs, result);
  result->reveal(out);
}

void test_fixedt_eq_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());

  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
  lhs->eq(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_eq_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
  lhs->eq(rhs, result);
  result->reveal(out);
}

void test_fixedt_neq_plain(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 4; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in[0], temp);

  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());

  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
  lhs->neq(in[1].get(), result);
  result->reveal(out);
}

void test_fixedt_neq_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  BooleanTensor<int64_t> *result =
      new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
  lhs->neq(rhs, result);
  result->reveal(out);
}

void test_fixedt_matmul_fixed(
    size_t p, std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
    TensorAdapter<int64_t> *out) {
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
  for (int i = 0; i < 6; i++) {
    temp.emplace_back(gen(out->shape()));
  }

  test_fixedt_gen_shares(p, in, temp);
  Fix64N16 *lhs = new Fix64N16(temp[0].get(), temp[1].get());
  Fix64N16 *rhs = new Fix64N16(temp[2].get(), temp[3].get());
  Fix64N16 *result = new Fix64N16(temp[4].get(), temp[5].get());
  lhs->mat_mul(rhs, result);
  result->reveal(out);
}

TEST_F(FixedTensorTest, matmulfixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {4.0, 4.0, 4.0, 4.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_matmul_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_matmul_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_matmul_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, share) {
  std::vector<size_t> shape = {2, 2};
  std::vector<float> in_val = {1.0, 1.0, 1.0, 1.0};
  PaddleTensor<int64_t> input =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(in_val, shape, _cpu_ctx);
  auto output = _s_tensor_factory->create<int64_t>(shape);

  // test_fixedt_share(0, &input, output.get());

  _t[0] = std::thread([this, &input, output]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_share(0, &input, output.get()); });

  });
  _t[1] = std::thread([this]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_share(1, nullptr, nullptr); });

  });
  _t[2] = std::thread([this]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_share(2, nullptr, nullptr); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(&input, output.get()));
}

TEST_F(FixedTensorTest, addfixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {3.0, 3.0, 3.0, 3.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_add_fixed(0, in, out0.get()); });
  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_add_fixed(1, in, out1.get()); });
  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_add_fixed(2, in, out2.get()); });
  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, addplain) {
  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {3.0, 3.0, 3.0, 3.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_add_plain(0, in, out0.get()); });
  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_add_plain(1, in, out1.get()); });
  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_add_plain(2, in, out2.get()); });
  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, subfixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 3.0, 3.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_sub_fixed(0, in, out0.get()); });
  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_sub_fixed(1, in, out1.get()); });
  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_sub_fixed(2, in, out2.get()); });
  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, subplain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 3.0, 3.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_sub_plain(0, in, out0.get()); });
  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_sub_plain(1, in, out1.get()); });
  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_sub_plain(2, in, out2.get()); });
  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, negfixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {-1.0, -1.0, -1.0, -1.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_neg_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_neg_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_neg_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, mulfixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_mul_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_mul_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_mul_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, mul2fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_mul2_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_mul2_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_mul2_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, mulplain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_mul_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_mul_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_mul_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, divplain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {0.5, 0.5, 0.5, 0.5};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_div_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_div_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_div_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, sum) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {4.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  std::vector<size_t> ret_shape = {1};
  auto out0 = _s_tensor_factory->create<int64_t>(ret_shape);
  auto out1 = _s_tensor_factory->create<int64_t>(ret_shape);
  auto out2 = _s_tensor_factory->create<int64_t>(ret_shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, ret_shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_sum_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_sum_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_sum_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, mat_mulfixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {4.0, 4.0, 4.0, 4.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_mat_mul_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_mat_mul_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_mat_mul_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, mat_mulplain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {4.0, 4.0, 4.0, 4.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_mat_mul_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_mat_mul_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_mat_mul_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, dot_mul_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {8.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  std::vector<size_t> ret_shape = {1};
  auto out0 = _s_tensor_factory->create<int64_t>(ret_shape);
  auto out1 = _s_tensor_factory->create<int64_t>(ret_shape);
  auto out2 = _s_tensor_factory->create<int64_t>(ret_shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, ret_shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_dot_mul_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_dot_mul_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_dot_mul_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, dot_mul_plain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {8.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  std::vector<size_t> ret_shape = {1};
  auto out0 = _s_tensor_factory->create<int64_t>(ret_shape);
  auto out1 = _s_tensor_factory->create<int64_t>(ret_shape);
  auto out2 = _s_tensor_factory->create<int64_t>(ret_shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, ret_shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_dot_mul_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_dot_mul_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_dot_mul_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, gt_plain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 3.0, 3.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 1 / pow(2, 16),
                                1 / pow(2, 16)};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_gt_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_gt_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_gt_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, gt_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 3.0, 3.0};
  std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 1 / pow(2, 16),
                                1 / pow(2, 16)};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_gt_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_gt_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_gt_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, lt_plain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {2.0, 2.0, 3.0, 3.0};
  std::vector<float> in1_val = {3.0, 3.0, 2.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_lt_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_lt_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_lt_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, lt_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {2.0, 2.0, 3.0, 3.0};
  std::vector<float> in1_val = {3.0, 3.0, 2.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_lt_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_lt_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_lt_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, leq_plain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {2.0, 3.0, 3.0, 3.0};
  std::vector<float> in1_val = {3.0, 3.0, 2.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_leq_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_leq_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_leq_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, leq_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {2.0, 3.0, 3.0, 3.0};
  std::vector<float> in1_val = {3.0, 3.0, 2.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_leq_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_leq_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_leq_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, geq_plain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 2.0, 2.0};
  std::vector<float> in1_val = {2.0, 3.0, 3.0, 3.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_geq_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_geq_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_geq_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, geq_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 2.0, 2.0};
  std::vector<float> in1_val = {2.0, 3.0, 3.0, 3.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_geq_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_geq_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_geq_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, eq_plain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 2.0, 3.0};
  std::vector<float> in1_val = {3.0, 3.0, 3.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_eq_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_eq_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_eq_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, eq_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 3.0, 2.0, 3.0};
  std::vector<float> in1_val = {3.0, 3.0, 3.0, 2.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_eq_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_eq_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_eq_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, neq_plain) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {2.0, 3.0, 3.0, 3.0};
  std::vector<float> in1_val = {3.0, 2.0, 3.0, 3.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_neq_plain(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_neq_plain(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_neq_plain(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, neq_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {3.0, 2.0, 3.0, 3.0};
  std::vector<float> in1_val = {2.0, 3.0, 3.0, 3.0};
  std::vector<float> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape),
                                                             gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val, shape, _cpu_ctx)
      .copy(in[1].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;
  dynamic_cast<PaddleTensor<int64_t> *>(in[1].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);
  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_neq_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_neq_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_neq_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, exp_fixed) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {0.0, 0.0, 1.0, 1.0};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {1.0, 1.0, 2.7183, 2.7183};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_exp_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_exp_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_exp_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.1));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.1));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.1));
}

TEST_F(FixedTensorTest, polynomial) {
  // y = 1 + x
  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {-1.0, 2.0, 2.0, 2.0};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {0.0, 3.0, 3.0, 3.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_poly_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_poly_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_poly_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, polynomial_wise) {
  // y = x + 1 (x >= 0)
  // y = 1 (x < 0)
  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {-1.0, 1.0, 2.0, 2.0};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {1.0, 2.0, 3.0, 3.0};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_poly_wise_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_poly_wise_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_poly_wise_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, relu) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, -1.0, -2, 2};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {1.0, 0.0, 0.0, 2};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_relu_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_relu_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_relu_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, softmax) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {1.0, 1.0, 1, 1};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {0.5, 0.5, 0.5, 0.5};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_softmax_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_softmax_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_softmax_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, sigmoid) {

  std::vector<size_t> shape = {2, 2};
  std::vector<float> in0_val = {0.0, 0.0, -0.5, 0.5};
  // std::vector<float> in1_val = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> res_val = {0.5, 0.5, 0.3775, 0.6225};
  std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

  test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val, shape, _cpu_ctx)
      .copy(in[0].get());
  // not copy scaling factor in copy funtion
  dynamic_cast<PaddleTensor<int64_t> *>(in[0].get())->scaling_factor() = 16;

  auto out0 = _s_tensor_factory->create<int64_t>(shape);
  auto out1 = _s_tensor_factory->create<int64_t>(shape);
  auto out2 = _s_tensor_factory->create<int64_t>(shape);

  PaddleTensor<int64_t> result =
      test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

  _t[0] = std::thread([this, in, out0]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[0],
        [&]() { test_fixedt_sigmoid_fixed(0, in, out0.get()); });

  });
  _t[1] = std::thread([this, in, out1]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[1],
        [&]() { test_fixedt_sigmoid_fixed(1, in, out1.get()); });

  });
  _t[2] = std::thread([this, in, out2]() mutable {
    g_ctx_holder::template run_with_context(
        _exec_ctx.get(), _mpc_ctx[2],
        [&]() { test_fixedt_sigmoid_fixed(2, in, out2.get()); });

  });

  _t[0].join();
  _t[1].join();
  _t[2].join();

  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.1));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.1));
  EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.1));
}

} // namespace aby3
