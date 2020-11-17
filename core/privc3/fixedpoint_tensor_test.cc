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

#include <string>
#include <cmath>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

#include "aby3_context.h"
#include "core/paddlefl_mpc/mpc_protocol/mesh_network.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "fixedpoint_tensor.h"

namespace aby3 {

using g_ctx_holder = paddle::mpc::ContextHolder;
using Fix64N16 = FixedPointTensor<int64_t, 16>;
using AbstractContext = paddle::mpc::AbstractContext;

class FixedTensorTest : public ::testing::Test {
public:

    paddle::platform::CPUDeviceContext _cpu_ctx;
    std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
    std::shared_ptr<AbstractContext> _mpc_ctx[3];
    std::shared_ptr<gloo::rendezvous::HashStore> _store;
    std::thread _t[3];
    std::shared_ptr<TensorAdapterFactory> _s_tensor_factory;

    virtual ~FixedTensorTest() noexcept {}

    void SetUp() {

        paddle::framework::OperatorBase* op = nullptr;
        paddle::framework::Scope scope;
        paddle::framework::RuntimeContext ctx({}, {});
        // only device_ctx is needed
        _exec_ctx = std::make_shared<paddle::framework::ExecutionContext>(
            *op, scope, _cpu_ctx, ctx);

        _store = std::make_shared<gloo::rendezvous::HashStore>();

        std::thread t[3];
        for (size_t i = 0; i < 3; ++i) {
            _t[i] = std::thread(&FixedTensorTest::gen_mpc_ctx, this, i);
        }
        for (auto& ti : _t) {
            ti.join();
        }
        _s_tensor_factory = std::make_shared<PaddleTensorFactory>(&_cpu_ctx);
    }
    std::shared_ptr<paddle::mpc::MeshNetwork> gen_network(size_t idx) {
        return std::make_shared<paddle::mpc::MeshNetwork>(idx,
                                                          "127.0.0.1",
                                                          3,
                                                          "test_prefix",
                                                          _store);
    }
    void gen_mpc_ctx(size_t idx) {
        auto net = gen_network(idx);
        net->init();
        _mpc_ctx[idx] = std::make_shared<ABY3Context>(idx, net);
    }

    std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
        return _s_tensor_factory->template create<int64_t>(shape);
    }
};

std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
    return g_ctx_holder::tensor_factory()->template create<int64_t>(shape);
}

template<typename T, size_t N>
PaddleTensor<T> test_fixedt_gen_paddle_tensor(std::vector<double>& input,
                        std::vector<size_t>& shape,
                        paddle::platform::CPUDeviceContext& cpu_ctx) {

    PaddleTensor<T> ret(&cpu_ctx);
    ret.reshape(shape);
    T* ret_ptr = ret.data();
    for (int i = 0; i < ret.numel(); i++) {
        *(ret_ptr + i) = (T) (input[i] * pow(2, N));
    }
    return ret;
}

template<typename T>
bool test_fixedt_check_tensor_eq(const TensorAdapter<T>* result,
                                 const TensorAdapter<T>* expected,
                                 double precision = 0.0001,
                                 bool use_relative_error = false) {
    // check shape
    std::vector<size_t> shape1, shape2;
    shape1 = result->shape();
    shape2 = expected->shape();
    size_t scale = result->scaling_factor();
    if (shape1.size() != shape2.size()) {
        std::cout << "shape size error: shape1.size: "<<shape1.size()<<
                     "; shape2.size: "<<shape2.size()<<std::endl;
        return false;
    }
    for (int i = 0; i < shape1.size(); i++) {
        if (shape1[i] != shape2[i]) {
            std::cout << "shape error!"<<std::endl;
            return false;
        }
    }

    // check each element
    bool return_false = false;
    for (int i = 0; i < result->numel(); i++) {
        // absolute error
        if (!use_relative_error && std::abs(*(result->data() + i) - *(expected->data() + i)) >
            precision * std::pow(2, scale)) {
            std::cout << "result error: index: "<< i <<
                        " output[i] = "<< *(result->data() + i) / pow(2, 16) <<
                        " expected[i] = " << *(expected->data() + i) / pow(2, 16) << std::endl;
            return_false = true;
        }
        // relative error
        if (use_relative_error
            && std::abs(*(result->data() + i) - *(expected->data() + i))
            / (std::abs(*(expected->data() + i))  + 0.00000001)
            > precision) {
            std::cout << "result error: index: "<< i <<
                        " output[i] = " << *(result->data() + i) / pow(2, 16) <<
                        " expected[i] = " << *(expected->data() + i) / pow(2, 16) << std::endl;
            return_false = true;
        }
    }
    if (return_false) return false;
    return true;
}

void test_fixedt_gen_shares(size_t p,
                std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
                std::vector<std::shared_ptr<TensorAdapter<int64_t>>>& out) {
    if (p == 0) {

        std::shared_ptr<TensorAdapter<int64_t>> out1_shared[3];
        std::shared_ptr<TensorAdapter<int64_t>> out2_shared[3];
        for (int i = 0; i < 3; i++) {
            out1_shared[i] = g_ctx_holder::tensor_factory()->
                                    create<int64_t>(out[0]->shape());
            out2_shared[i] = g_ctx_holder::tensor_factory()->
                                    create<int64_t>(out[0]->shape());
        }
        TensorAdapter<int64_t>* out1[3] = {out1_shared[0].get(),
                            out1_shared[1].get(), out1_shared[2].get()};
        TensorAdapter<int64_t>* out2[3] = {out2_shared[0].get(),
                            out2_shared[1].get(), out2_shared[2].get()};

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
            out3_shared[i] = g_ctx_holder::tensor_factory()->
                                    create<int64_t>(out[0]->shape());
        }
        TensorAdapter<int64_t>* out3[4] = {out3_shared[0].get(),
                                           out3_shared[1].get(),
                                           out3_shared[2].get(),
                                           out3_shared[3].get()};
        for (int i = 0; i < 4; i++) {
            g_ctx_holder::mpc_ctx()->network()->template recv(0, *out3[i]);
            out3[i]->copy(out[i].get());
        }
    }

}

void test_fixedt_gen_shares(size_t p,
                std::shared_ptr<TensorAdapter<int64_t>> in,
                std::vector<std::shared_ptr<TensorAdapter<int64_t>>>& out) {

    if (p == 0) {

        std::shared_ptr<TensorAdapter<int64_t>> out1_shared[3];
        for (int i = 0; i < 3; i++) {
            out1_shared[i] = g_ctx_holder::tensor_factory()->
                                create<int64_t>(out[0]->shape());
        }
        TensorAdapter<int64_t>* out1[3] = {out1_shared[0].get(),
                            out1_shared[1].get(), out1_shared[2].get()};
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
            out3_shared[i] = g_ctx_holder::tensor_factory()->
                                create<int64_t>(out[0]->shape());
        }
        TensorAdapter<int64_t>* out3[2] = {out3_shared[0].get(),
                            out3_shared[1].get()};
        for (int i = 0; i < 2; i++) {
            g_ctx_holder::mpc_ctx()->network()->template recv(0, *out3[i]);
            out3[i]->copy(out[i].get());
        }
    }

}

void test_fixedt_share(size_t p, TensorAdapter<int64_t>* in,
               TensorAdapter<int64_t>* ret) {

    if (in || ret) {
        TensorAdapter<int64_t>* output[3];
        for (int i = 0; i < 3; i++) {
            output[i] = new PaddleTensor<int64_t>(g_ctx_holder::device_ctx());
            dynamic_cast<PaddleTensor<int64_t>*>(output[i])->reshape(in->shape());
        }
        Fix64N16::share(in, output);
        output[0]->add(output[1], ret);
        ret->add(output[2], ret);

        for (int i = 0; i < 3; i++) {
            delete output[i];
        }
    }
}

void test_fixedt_add_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    Fix64N16* result = new Fix64N16(temp[4].get(), temp[5].get());
    lhs->add(rhs, result);
    result->reveal(out);
}

void test_fixedt_add_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->add(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_sub_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    Fix64N16* result = new Fix64N16(temp[4].get(), temp[5].get());
    lhs->sub(rhs, result);
    result->reveal(out);
}

void test_fixedt_sub_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->sub(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_neg_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->negative(result);
    result->reveal(out);
}

void test_fixedt_mul_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    Fix64N16* result = new Fix64N16(temp[4].get(), temp[5].get());
    lhs->mul(rhs, result);
    result->reveal(out);
}

void test_fixedt_mul_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->mul(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_div_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->div(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_div_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    Fix64N16* result = new Fix64N16(temp[4].get(), temp[5].get());
    lhs->div(rhs, result);
    result->reveal(out);
}

void test_fixedt_sum_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 2; i++) {
        temp.emplace_back(gen(in[0]->shape()));
    }

    for (int i = 2; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->sum(result);

    result->reveal(out);
}

void test_fixedt_poly_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    auto c_shape = out->shape();
    c_shape.insert(c_shape.begin(), 2);
    //construct coeff
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

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->polynomial(coeff.get(), result);
    result->reveal(out);
}

void test_fixedt_poly_wise_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());

    //constrct break_point
    auto shape = in[0]->shape();
    shape.insert(shape.begin(), 1);
    auto break_point = gen(shape);
    for (size_t i = 0; i < break_point->numel(); ++i) {
        break_point->data()[i] = 0;
    }
    break_point->scaling_factor() = 16;

    //contruct coeff
    std::vector<size_t> shape_ = {2, 2};
    auto in_shape = in[0]->shape();
    shape_.insert(shape_.end(), in_shape.begin(), in_shape.end());
    auto coeff = gen(shape_);
    int64_t* c_ptr = coeff->data();
    for (size_t i = 0; i < 4 * in[0]->numel(); i++) {
        *(c_ptr + i) = 1 << 16;
    }
    for (size_t i = in[0]->numel(); i < in[0]->numel() * 2; i++) {
        *(c_ptr + i) = 0;
    }
    coeff->scaling_factor() = 16;

    lhs->polynomial_piecewise(coeff.get(), break_point.get(), result);

    result->reveal(out);
}

void test_fixedt_relu_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->relu(result);
    result->reveal(out);
}

void test_fixedt_relu2_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->relu_with_derivative(result, nullptr);
    result->reveal(out);
}

void test_fixedt_softmax_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->softmax(result);
    result->reveal(out);
}

void test_fixedt_sigmoid_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->sigmoid(result);
    result->reveal(out);
}

void test_fixedt_sigmoid_enhanced_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->sigmoid_enhanced(result);
    result->reveal(out);
}

void test_fixedt_sigmoid_chebyshev_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->sigmoid_chebyshev(result);
    result->reveal(out);
}

void test_fixedt_sigmoid_high_precision_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->sigmoid_high_precision(result);
    result->reveal(out);
}

void test_fixedt_exp_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->exp(result);
    result->reveal(out);
}

void test_fixedt_mat_mul_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->mat_mul(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_dot_mul_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(in[0]->shape()));
    }
    for (int i = 4; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    Fix64N16* result = new Fix64N16(temp[4].get(), temp[5].get());
    lhs->dot_mul(rhs, result);
    result->reveal(out);
}

void test_fixedt_dot_mul_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 2; i++) {
        temp.emplace_back(gen(in[0]->shape()));
    }
    for (int i = 2; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* result = new Fix64N16(temp[2].get(), temp[3].get());
    lhs->dot_mul(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_gt_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());

    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
    lhs->gt(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_gt_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
    lhs->gt(rhs, result);
    result->reveal(out);
}

void test_fixedt_lt_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());

    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
    lhs->lt(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_lt_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
    lhs->lt(rhs, result);
    result->reveal(out);
}

void test_fixedt_leq_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());

    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
    lhs->leq(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_leq_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
    lhs->leq(rhs, result);
    result->reveal(out);
}

void test_fixedt_geq_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());

    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
    lhs->geq(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_geq_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
    lhs->geq(rhs, result);
    result->reveal(out);
}

void test_fixedt_eq_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());

    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
    lhs->eq(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_eq_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
    lhs->eq(rhs, result);
    result->reveal(out);
}

void test_fixedt_neq_plain(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 4; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in[0], temp);

    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());

    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[2].get(), temp[3].get());
    lhs->neq(in[1].get(), result);
    result->reveal(out);
}

void test_fixedt_neq_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    BooleanTensor<int64_t>* result =
        new BooleanTensor<int64_t>(temp[4].get(), temp[5].get());
    lhs->neq(rhs, result);
    result->reveal(out);
}

void test_fixedt_matmul_fixed(size_t p,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    for (int i = 0; i < 2; i++) {
        temp.emplace_back(gen(in[0]->shape()));
    }
    for (int i = 2; i < 4; i++) {
        temp.emplace_back(gen(in[1]->shape()));
    }
    for (int i = 4; i < 6; i++) {
        temp.emplace_back(gen(out->shape()));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* lhs = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* rhs = new Fix64N16(temp[2].get(), temp[3].get());
    Fix64N16* result = new Fix64N16(temp[4].get(), temp[5].get());
    lhs->mat_mul(rhs, result);
    result->reveal(out);
}

void test_fixedt_precision_recall_fixed(size_t p,
               double threshold,
               std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in,
               TensorAdapter<int64_t>* out) {
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> temp;
    // preds
    for (int i = 0; i < 2; i++) {
        temp.emplace_back(gen(in[0]->shape()));
    }
    // labels
    for (int i = 0; i < 2; i++) {
        temp.emplace_back(gen(in[1]->shape()));
    }
    // indices
    for (int i = 0; i < 2; i++) {
        temp.emplace_back(gen(in[0]->shape()));
    }
    std::vector<size_t> shape_ = {3};
    // tp fp fn
    for (int i = 0; i < 2; i++) {
        temp.emplace_back(gen(shape_));
    }

    test_fixedt_gen_shares(p, in, temp);
    Fix64N16* preds   = new Fix64N16(temp[0].get(), temp[1].get());
    Fix64N16* labels  = new Fix64N16(temp[2].get(), temp[3].get());
    Fix64N16* indices = new Fix64N16(temp[4].get(), temp[5].get());
    Fix64N16* tpfpfn  = new Fix64N16(temp[6].get(), temp[7].get());

    Fix64N16::preds_to_indices(preds, indices, threshold);
    Fix64N16::calc_tp_fp_fn(indices, labels, tpfpfn);
    Fix64N16::calc_precision_recall(tpfpfn, out);
}

TEST_F(FixedTensorTest, matmulfixed) {

    std::vector<size_t> shape = {1, 3};
    std::vector<size_t> shape1 = {3, 1};
    std::vector<size_t> shape_o = {1, 1};
    std::vector<double> in0_val = {1, 0, 0};
    std::vector<double> in1_val = {1, 2, 3};
    std::vector<double> res_val = {1};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape1)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape1, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape_o);
    auto out1 = _s_tensor_factory->create<int64_t>(shape_o);
    auto out2 = _s_tensor_factory->create<int64_t>(shape_o);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape_o, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_matmul_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_matmul_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_matmul_fixed(2, in, out2.get());
        });

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
    std::vector<double> in_val = {1.0, 1.0, 1.0, 1.0};
    PaddleTensor<int64_t> input =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(in_val, shape, _cpu_ctx);
    auto output = _s_tensor_factory->create<int64_t>(shape);

    //test_fixedt_share(0, &input, output.get());

    _t[0] = std::thread([this, &input, output]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_share(0, &input, output.get());
        });

    });
    _t[1] = std::thread([this]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_share(1, nullptr, nullptr);
        });

    });
    _t[2] = std::thread([this]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_share(2, nullptr, nullptr);
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(&input, output.get()));

}


TEST_F(FixedTensorTest, addfixed) {

    std::vector<size_t> shape = {2, 2};
    std::vector<double> in0_val = {0x1p47 - 1, 5+2^-16, 1.0, 1.0};
    std::vector<double> in1_val = {1.0, 8+(1-2^-16), 2.0, 2.0};
    std::vector<double> res_val = {-0x1p47, 14, 3.0, 3.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_add_fixed(0, in, out0.get());
        });
    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_add_fixed(1, in, out1.get());
        });
    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_add_fixed(2, in, out2.get());
        });
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
    std::vector<double> in0_val = {1.0, 5+2^-16, 1.0, 1.0};
    std::vector<double> in1_val = {0x1p47 - 1, 8+(1-2^-16), 2.0, 2.0};
    std::vector<double> res_val = {-0x1p47, 14.0, 3.0, 3.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_add_plain(0, in, out0.get());
        });
    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_add_plain(1, in, out1.get());
        });
    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_add_plain(2, in, out2.get());
        });
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
    std::vector<double> in0_val = {3.0, 3.0, 3.0, 3.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_sub_fixed(0, in, out0.get());
        });
    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_sub_fixed(1, in, out1.get());
        });
    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_sub_fixed(2, in, out2.get());
        });
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
    std::vector<double> in0_val = {3.0, 3.0, 3.0, 3.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_sub_plain(0, in, out0.get());
        });
    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_sub_plain(1, in, out1.get());
        });
    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_sub_plain(2, in, out2.get());
        });
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
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    //std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {-1.0, -1.0, -1.0, -1.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_neg_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_neg_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_neg_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {2.0, 2.0, 2.0, 2.0};

    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_mul_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_mul_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_mul_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

#ifndef USE_ABY3_TRUNC1 //use aby3 trunc1
TEST_F(FixedTensorTest, mulfixed_multi_times) {

    std::vector<size_t> shape = {100000, 1};
    std::vector<double> in0_val(shape[0]), in1_val(shape[0]), res_val(shape[0]);

    auto fill_mul_data = [&in0_val, &in1_val, &res_val] () {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);

        std::uniform_int_distribution<int64_t> input(-0x1p36, 0x1p36);
        std::for_each(in0_val.begin(), in0_val.end(),
                        [] (double& a){ a = 1.0;});
        std::for_each(in1_val.begin(), in1_val.end(),
                        [&input, &generator] (double& a){ a = input(generator) * pow(2, -16);});
        std::transform(in0_val.begin(), in0_val.end(), in1_val.begin(), res_val.begin(),
                        [] (double& a, double& b){ return a * b;});
        };
    fill_mul_data();
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_mul_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_mul_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_mul_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}
#endif

TEST_F(FixedTensorTest, mulfixed_overflow) {

    std::vector<size_t> shape = {1};
    // result greater than 2^32 is overflow
    // notice: multiplier larger than 2^20 may lead to error result
    // as 2^l << 2^k [ stated in ABY3]
    std::vector<double> in0_val = {0x1p16};
    std::vector<double> in1_val = {0x1p16};
    std::vector<double> res_val = {0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_mul_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_mul_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_mul_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, mulfixed_upper_bound) {

    std::vector<size_t> shape = {1, 2};
    // recommend each input less than 2^20
    // larger than 2^20 may lead to error result
    // as 2^l << 2^k [stated in ABY3]
    std::vector<double> in0_val = {1.0, 1.0};
    std::vector<double> in1_val = {0x1p20, -0x1p20};
    std::vector<double> res_val = {0x1p20, -0x1p20};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_mul_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_mul_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_mul_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, mulfixed_low_bound) {

    std::vector<size_t> shape = {1};
    std::vector<double> in0_val = {1.0};
    std::vector<double> in1_val = {0x1p-16};
    std::vector<double> res_val = {0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_mul_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_mul_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_mul_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_mul_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_mul_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_mul_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {0.5, 0.5, 0.5, 0.5};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_div_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_div_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_div_plain(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, divfixed) {

    std::vector<size_t> shape = {2, 2};
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> in1_val = {1.0, 10.0, 1000.0, 700.0};
    std::vector<double> res_val = {1.0, 0.1, 0.001, 1.0 / 700};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_div_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_div_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_div_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.2, true));
}

TEST_F(FixedTensorTest, divfixed_low_bound) {

    std::vector<size_t> shape = {1};
    std::vector<double> in0_val = {1.0};
    // divisor > 1/x0, default x0 = 2^-15
    std::vector<double> in1_val = {0x1p15};
    std::vector<double> res_val = {0x1p-15};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_div_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_div_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_div_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.0001, true));
}

TEST_F(FixedTensorTest, sum) {

    std::vector<size_t> shape = {2, 2};
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    //std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {4.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    std::vector<size_t> ret_shape = {1};
    auto out0 = _s_tensor_factory->create<int64_t>(ret_shape);
    auto out1 = _s_tensor_factory->create<int64_t>(ret_shape);
    auto out2 = _s_tensor_factory->create<int64_t>(ret_shape);



    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, ret_shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_sum_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_sum_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_sum_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {4.0, 4.0, 4.0, 4.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_mat_mul_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_mat_mul_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_mat_mul_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {8.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    std::vector<size_t> ret_shape = {1};
    auto out0 = _s_tensor_factory->create<int64_t>(ret_shape);
    auto out1 = _s_tensor_factory->create<int64_t>(ret_shape);
    auto out2 = _s_tensor_factory->create<int64_t>(ret_shape);



    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, ret_shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_dot_mul_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_dot_mul_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_dot_mul_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {8.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    std::vector<size_t> ret_shape = {1};
    auto out0 = _s_tensor_factory->create<int64_t>(ret_shape);
    auto out1 = _s_tensor_factory->create<int64_t>(ret_shape);
    auto out2 = _s_tensor_factory->create<int64_t>(ret_shape);



    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, ret_shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_dot_mul_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_dot_mul_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_dot_mul_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {3.0, 3.0, 3.0, 3.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 1 / pow(2, 16), 1 / pow(2, 16)};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_gt_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_gt_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_gt_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {3.0, 3.0, 3.0, 3.0};
    std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 1 / pow(2, 16), 1 / pow(2, 16)};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_gt_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_gt_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_gt_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {2.0, 2.0, 3.0, 3.0};
    std::vector<double> in1_val = {3.0, 3.0, 2.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_lt_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_lt_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_lt_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {2.0, 2.0, 3.0, 3.0};
    std::vector<double> in1_val = {3.0, 3.0, 2.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_lt_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_lt_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_lt_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {2.0, 3.0, 3.0, 3.0};
    std::vector<double> in1_val = {3.0, 3.0, 2.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_leq_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_leq_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_leq_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {2.0, 3.0, 3.0, 3.0};
    std::vector<double> in1_val = {3.0, 3.0, 2.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_leq_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_leq_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_leq_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {3.0, 3.0, 2.0, 2.0};
    std::vector<double> in1_val = {2.0, 3.0, 3.0, 3.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_geq_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_geq_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_geq_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {3.0, 3.0, 2.0, 2.0};
    std::vector<double> in1_val = {2.0, 3.0, 3.0, 3.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_geq_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_geq_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_geq_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {3.0, 3.0, 2.0, 3.0};
    std::vector<double> in1_val = {3.0, 3.0, 3.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_eq_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_eq_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_eq_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {3.0, 3.0, 2.0, 3.0};
    std::vector<double> in1_val = {3.0, 3.0, 3.0, 2.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_eq_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_eq_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_eq_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {2.0, 3.0, 3.0, 3.0};
    std::vector<double> in1_val = {3.0, 2.0, 3.0, 3.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_neq_plain(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_neq_plain(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_neq_plain(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {3.0, 2.0, 3.0, 3.0};
    std::vector<double> in1_val = {2.0, 3.0, 3.0, 3.0};
    std::vector<double> res_val = {1 / pow(2, 16), 1 / pow(2, 16), 0, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;
    dynamic_cast<PaddleTensor<int64_t>*>(in[1].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);
    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_neq_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_neq_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_neq_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {0.0, 0.0, 1.0, 1.0};
    std::vector<double> res_val = {1.0, 1.0, 2.71828, 2.71828};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_exp_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_exp_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_exp_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.01, true));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.01, true));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.01, true));
}

TEST_F(FixedTensorTest, exp_fixed_low_bound) {

    std::vector<size_t> shape = {1, 3};
    // exp approximate: exp(x) = \lim_{n->inf} (1+x/n)^n
    // where n = 2^ite = 256, therefore, exp(-512) = exp(0),
    // exp(-511) = exp(-1), exp(-256) = 0
    std::vector<double> in0_val = {-512, -511, -256};
    std::vector<double> res_val = {1, 0.367879, 0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_exp_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_exp_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_exp_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.01, true));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.01, true));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.01, true));
}

TEST_F(FixedTensorTest, exp_fixed_upper_bound) {
    std::vector<size_t> shape = {1};
    // input large than 15 may get error result because of multiplication error
    std::vector<double> in0_val = {15};
    std::vector<double> res_val = {3269017.37};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_exp_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_exp_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_exp_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.4, true));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.4, true));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.4, true));
}

TEST_F(FixedTensorTest, polynomial) {
    // y = 1 + x
    std::vector<size_t> shape = {2, 2};
    std::vector<double> in0_val = {-1.0, 2.0, 2.0, 2.0};
    //std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {0.0, 3.0, 3.0, 3.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_poly_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_poly_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_poly_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {-1.0, 1.0, 2.0, 2.0};
    //std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {1.0, 2.0, 3.0, 3.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_poly_wise_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_poly_wise_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_poly_wise_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, -1.0, -2, 2};
    //std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {1.0, 0.0, 0.0, 2};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_relu_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_relu_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_relu_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, relu_low_bound) {

    std::vector<size_t> shape = {1};
    std::vector<double> in0_val = {-0x1p-20};
    std::vector<double> res_val = {0.0};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_relu_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_relu_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_relu_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, relu_upper_bound) {

    std::vector<size_t> shape = {1};
    std::vector<double> in0_val = {0x1p20};
    std::vector<double> res_val = {0x1p20};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_relu_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_relu_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_relu_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

TEST_F(FixedTensorTest, relu2) {

    std::vector<size_t> shape = {2, 2};
    std::vector<double> in0_val = {1.0, -1.0, -2, 2};
    //std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {1.0, 0.0, 0.0, 2};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_relu2_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_relu2_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_relu2_fixed(2, in, out2.get());
        });

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
    std::vector<double> in0_val = {1.0, 1.0, 1, 1};
    //std::vector<double> in1_val = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> res_val = {0.5, 0.5, 0.5, 0.5};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_softmax_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_softmax_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_softmax_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.1));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.1));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.1));
}

TEST_F(FixedTensorTest, sigmoid_chebyshev) {

    std::vector<size_t> shape = {2, 2};
    // larger error when input < -3 or >4
    std::vector<double> in0_val = {1.0, 2.0, -3.0, 4.0};
    std::vector<double> res_val = {0.73105, 0.88079, 0.0474, 0.9820};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_sigmoid_chebyshev_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_sigmoid_chebyshev_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_sigmoid_chebyshev_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.03));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.03));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.03));
}

TEST_F(FixedTensorTest, sigmoid_high_precision) {

    std::vector<size_t> shape = {2, 2};
    // larger error when input < -3 or >4
    std::vector<double> in0_val = {1.0, 2.0, -3.0, 4.0};
    std::vector<double> res_val = {0.73105, 0.88079, 0.0474, 0.9820};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_sigmoid_high_precision_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_sigmoid_high_precision_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_sigmoid_high_precision_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.003));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.003));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.003));
}

TEST_F(FixedTensorTest, sigmoid) {

    std::vector<size_t> shape = {2, 2};
    std::vector<double> in0_val = {0.0, 3, 7, 0.5};
    std::vector<double> res_val = {0.5, 0.9525, 0.999, 0.6225};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_sigmoid_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_sigmoid_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_sigmoid_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.08));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.08));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.08));
}

TEST_F(FixedTensorTest, sigmoid_enhanced) {

    std::vector<size_t> shape = {2, 2};
    std::vector<double> in0_val = {0.0, 3, 7, 0.5};
    std::vector<double> res_val = {0.5, 0.9525, 0.999, 0.6225};
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in = {gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    //not copy scaling factor in copy funtion
    dynamic_cast<PaddleTensor<int64_t>*>(in[0].get())->
                                scaling_factor() = 16;

    auto out0 = _s_tensor_factory->create<int64_t>(shape);
    auto out1 = _s_tensor_factory->create<int64_t>(shape);
    auto out2 = _s_tensor_factory->create<int64_t>(shape);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape, _cpu_ctx);

    _t[0] = std::thread([this, in, out0]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_sigmoid_enhanced_fixed(0, in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_sigmoid_enhanced_fixed(1, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_sigmoid_enhanced_fixed(2, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get(), 0.08));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get(), 0.08));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result, 0.08));
}

TEST_F(FixedTensorTest, max_test) {
    std::vector<size_t> shape = { 1 };
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen(shape), gen(shape), gen(shape) };
    std::shared_ptr<TensorAdapter<int64_t>> sr[3] = { gen(shape), gen(shape), gen(shape) };

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen(shape), gen(shape), gen(shape),
                                                        gen(shape), gen(shape), gen(shape)};

    std::shared_ptr<TensorAdapter<int64_t>> sbout[6] = {
        gen(shape), gen(shape), gen(shape), gen(shape), gen(shape), gen(shape)};

    // lhs = 6 = 1 + 2 + 3
    sl[0]->data()[0] = 1;
    sl[1]->data()[0] = 2;
    sl[2]->data()[0] = 3;
    // rhs = 15 = 4 + 5 + 6
    sr[0]->data()[0] = 4;
    sr[1]->data()[0] = 5;
    sr[2]->data()[0] = 6;
    Fix64N16 fl0(sl[0].get(), sl[1].get());
    Fix64N16 fl1(sl[1].get(), sl[2].get());
    Fix64N16 fl2(sl[2].get(), sl[0].get());
    Fix64N16 fr0(sr[0].get(), sr[1].get());
    Fix64N16 fr1(sr[1].get(), sr[2].get());
    Fix64N16 fr2(sr[2].get(), sr[0].get());
    Fix64N16 fout0(sout[0].get(), sout[1].get());
    Fix64N16 fout1(sout[2].get(), sout[3].get());
    Fix64N16 fout2(sout[4].get(), sout[5].get());
    BooleanTensor<int64_t> bout0(sbout[0].get(), sbout[1].get());
    BooleanTensor<int64_t> bout1(sbout[2].get(), sbout[3].get());
    BooleanTensor<int64_t> bout2(sbout[4].get(), sbout[5].get());

    auto p = gen(shape);
    auto pb = gen(shape);

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                fl0.max(&fr0, &fout0, &bout0);
                fout0.reveal_to_one(0, p.get());
                bout0.reveal_to_one(0, pb.get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                fl1.max(&fr1, &fout1, &bout1);
                fout1.reveal_to_one(0, nullptr);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );
    _t[2] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                fl2.max(&fr2, &fout2, &bout2);
                fout2.reveal_to_one(0, nullptr);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(std::max(6, 15), p->data()[0]);
    EXPECT_EQ(1, pb->data()[0]);
}

TEST_F(FixedTensorTest, max_test2) {
    std::vector<size_t> shape = { 1 };
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen(shape), gen(shape), gen(shape) };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen(shape), gen(shape), gen(shape),
                                                        gen(shape), gen(shape), gen(shape)};
    // lhs = 6 = 1 + 2 + 3
    sl[0]->data()[0] = 1 << 16;
    sl[1]->data()[0] = 2 << 16;
    sl[2]->data()[0] = 3 << 16;

    auto pr = gen(shape);

    // rhs = 15
    pr->data()[0] = 15 << 16;
    pr->scaling_factor() = 16;
    Fix64N16 fl0(sl[0].get(), sl[1].get());
    Fix64N16 fl1(sl[1].get(), sl[2].get());
    Fix64N16 fl2(sl[2].get(), sl[0].get());
    Fix64N16 fout0(sout[0].get(), sout[1].get());
    Fix64N16 fout1(sout[2].get(), sout[3].get());
    Fix64N16 fout2(sout[4].get(), sout[5].get());

    auto p = gen(shape);

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                fl0.max(pr.get(), &fout0);
                fout0.reveal_to_one(0, p.get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                fl1.max(pr.get(), &fout1);
                fout1.reveal_to_one(0, nullptr);
            });
        }
    );
    _t[2] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                fl2.max(pr.get(), &fout2);
                fout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(std::max(6, 15), p->data()[0] >> 16);
}

TEST_F(FixedTensorTest, max_pooling_test) {
    std::vector<size_t> shape = { 4, 1 };
    std::vector<size_t> shape_ = { 1, 1 };

    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen(shape), gen(shape), gen(shape) };
    std::shared_ptr<TensorAdapter<int64_t>> sfout[6] = {
        gen(shape_), gen(shape_), gen(shape_), gen(shape_), gen(shape_), gen(shape_)};
    std::shared_ptr<TensorAdapter<int64_t>> sbout[6] = {
        gen(shape), gen(shape), gen(shape), gen(shape), gen(shape), gen(shape)};

    assign_to_tensor(sl[1].get(), 0l);
    assign_to_tensor(sl[2].get(), 0l);
    sl[0]->data()[0] = 2;
    sl[0]->data()[1] = 1;
    sl[0]->data()[2] = 4;
    sl[0]->data()[3] = 3;
    // input [2 1 4 3]

    auto pmax = gen(shape_);
    auto ppos = gen(shape);

    Fix64N16 fl0(sl[0].get(), sl[1].get());
    Fix64N16 fl1(sl[1].get(), sl[2].get());
    Fix64N16 fl2(sl[2].get(), sl[0].get());

    Fix64N16 fout0(sfout[0].get(), sfout[1].get());
    Fix64N16 fout1(sfout[2].get(), sfout[3].get());
    Fix64N16 fout2(sfout[4].get(), sfout[5].get());

    BooleanTensor<int64_t> bout0(sbout[0].get(), sbout[1].get());
    BooleanTensor<int64_t> bout1(sbout[2].get(), sbout[3].get());
    BooleanTensor<int64_t> bout2(sbout[4].get(), sbout[5].get());

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                fl0.max_pooling(&fout0, &bout0);
                fout0.reveal_to_one(0, pmax.get());
                bout0.reveal_to_one(0, ppos.get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                fl1.max_pooling(&fout1, &bout1);
                fout1.reveal_to_one(0, nullptr);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );
    _t[2] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                fl2.max_pooling(&fout2, &bout2);
                fout2.reveal_to_one(0, nullptr);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(4, pmax->data()[0]);

    EXPECT_EQ(0, ppos->data()[0]);
    EXPECT_EQ(0, ppos->data()[1]);
    EXPECT_EQ(1, ppos->data()[2]);
    EXPECT_EQ(0, ppos->data()[3]);
}

TEST_F(FixedTensorTest, inv_sqrt_test) {
    std::vector<size_t> shape = { 1 };

    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen(shape), gen(shape), gen(shape) };
    std::shared_ptr<TensorAdapter<int64_t>> sfout[6] = {
        gen(shape), gen(shape), gen(shape), gen(shape), gen(shape), gen(shape)};

    sl[0]->data()[0] = 0x4p16;
    sl[1]->data()[0] = 0;
    sl[2]->data()[0] = 0;
    // input [4]

    auto p = gen(shape);

    Fix64N16 fl0(sl[0].get(), sl[1].get());
    Fix64N16 fl1(sl[1].get(), sl[2].get());
    Fix64N16 fl2(sl[2].get(), sl[0].get());

    Fix64N16 fout0(sfout[0].get(), sfout[1].get());
    Fix64N16 fout1(sfout[2].get(), sfout[3].get());
    Fix64N16 fout2(sfout[4].get(), sfout[5].get());

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                fl0.inverse_square_root(&fout0);
                fout0.reveal_to_one(0, p.get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                fl1.inverse_square_root(&fout1);
                fout1.reveal_to_one(0, nullptr);
            });
        }
    );
    _t[2] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                fl2.inverse_square_root(&fout2);
                fout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    // inv_sqrt(4) = 1/2
    EXPECT_NEAR(0.5, p->data()[0] / 0x1p16f, 2 / 0x1p16f);

}

#ifdef USE_ABY3_TRUNC1 //use aby3 trunc1
TEST_F(FixedTensorTest, truncate1_msb_incorrect) {
    std::vector<size_t> shape = { 1 };
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen(shape), gen(shape), gen(shape) };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen(shape), gen(shape), gen(shape),
                                                        gen(shape), gen(shape), gen(shape)};
    // lhs = 6 = 1 + 2 + 3, share before truncate
    // zero share 0 = (1 << 62) + (1 << 62) - (1 << 63)
    sl[0]->data()[0] = ((int64_t) 3 << 32) - ((uint64_t) 1 << 63);
    sl[1]->data()[0] = ((int64_t) 2 << 32) + ((int64_t) 1 << 62);
    sl[2]->data()[0] = ((int64_t) 1 << 32) + ((int64_t) 1 << 62);

    auto pr = gen(shape);

    // rhs = 15
    pr->data()[0] = 6 << 16;
    pr->scaling_factor() = 16;
    Fix64N16 fl0(sl[0].get(), sl[1].get());
    Fix64N16 fl1(sl[1].get(), sl[2].get());
    Fix64N16 fl2(sl[2].get(), sl[0].get());
    Fix64N16 fout0(sout[0].get(), sout[1].get());
    Fix64N16 fout1(sout[2].get(), sout[3].get());
    Fix64N16 fout2(sout[4].get(), sout[5].get());

    auto p = gen(shape);

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                Fix64N16::truncate(&fl0, &fout0, 16);
                fout0.reveal_to_one(0, p.get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                Fix64N16::truncate(&fl1, &fout1, 16);
                fout1.reveal_to_one(0, nullptr);
            });
        }
    );
    _t[2] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                Fix64N16::truncate(&fl2, &fout2, 16);
                fout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    // failed: result is not close to 6
    EXPECT_GT(std::abs((p->data()[0] >> 16) - 6), 1000);
}

#else
TEST_F(FixedTensorTest, truncate3_msb_correct) {
    std::vector<size_t> shape = { 1 };
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen(shape), gen(shape), gen(shape) };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen(shape), gen(shape), gen(shape),
                                                        gen(shape), gen(shape), gen(shape)};
    // lhs = 6 = 1 + 2 + 3, share before truncate
    // zero share 0 = (1 << 62) + (1 << 62) - (1 << 63)
    sl[0]->data()[0] = ((int64_t) 3 << 32) - ((uint64_t) 1 << 63);
    sl[1]->data()[0] = ((int64_t) 2 << 32) + ((int64_t) 1 << 62);
    sl[2]->data()[0] = ((int64_t) 1 << 32) + ((int64_t) 1 << 62);

    auto pr = gen(shape);

    // rhs = 15
    pr->data()[0] = 6 << 16;
    pr->scaling_factor() = 16;
    Fix64N16 fl0(sl[0].get(), sl[1].get());
    Fix64N16 fl1(sl[1].get(), sl[2].get());
    Fix64N16 fl2(sl[2].get(), sl[0].get());
    Fix64N16 fout0(sout[0].get(), sout[1].get());
    Fix64N16 fout1(sout[2].get(), sout[3].get());
    Fix64N16 fout2(sout[4].get(), sout[5].get());

    auto p = gen(shape);

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                Fix64N16::truncate(&fl0, &fout0, 16);
                fout0.reveal_to_one(0, p.get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                Fix64N16::truncate(&fl1, &fout1, 16);
                fout1.reveal_to_one(0, nullptr);
            });
        }
    );
    _t[2] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                Fix64N16::truncate(&fl2, &fout2, 16);
                fout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ((p->data()[0] >> 16), 6);
}
#endif

TEST_F(FixedTensorTest, precision_recall) {

    std::vector<size_t> shape = {6};
    std::vector<size_t> shape_o = {3};
    std::vector<double> in0_val = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    std::vector<double> in1_val = {0, 1, 0, 1, 0 ,1};
    std::vector<double> res_val = {0.5, 1.0/3, 0.4};
    double threshold = 0.7;
    std::vector<std::shared_ptr<TensorAdapter<int64_t>>> in =
                            {gen(shape), gen(shape)};

    test_fixedt_gen_paddle_tensor<int64_t, 16>(in0_val,
                                shape, _cpu_ctx).copy(in[0].get());
    test_fixedt_gen_paddle_tensor<int64_t, 16>(in1_val,
                                shape, _cpu_ctx).copy(in[1].get());

    auto out0 = _s_tensor_factory->create<int64_t>(shape_o);
    auto out1 = _s_tensor_factory->create<int64_t>(shape_o);
    auto out2 = _s_tensor_factory->create<int64_t>(shape_o);

    PaddleTensor<int64_t> result =
            test_fixedt_gen_paddle_tensor<int64_t, 16>(res_val, shape_o, _cpu_ctx);

    _t[0] = std::thread([this, in, out0, threshold]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [&](){
            test_fixedt_precision_recall_fixed(0, threshold,  in, out0.get());
        });

    });
    _t[1] = std::thread([this, in, out1, threshold]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[1], [&](){
            test_fixedt_precision_recall_fixed(1, threshold, in, out1.get());
        });

    });
    _t[2] = std::thread([this, in, out2, threshold]() mutable {
        g_ctx_holder::template run_with_context(_exec_ctx.get(), _mpc_ctx[2], [&](){
            test_fixedt_precision_recall_fixed(2, threshold, in, out2.get());
        });

    });

    _t[0].join();
    _t[1].join();
    _t[2].join();

    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), out1.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out1.get(), out2.get()));
    EXPECT_TRUE(test_fixedt_check_tensor_eq(out0.get(), &result));
}

} // namespace aby3
