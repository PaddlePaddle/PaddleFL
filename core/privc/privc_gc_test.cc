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

#include "./privc_context.h"
#include "core/paddlefl_mpc/mpc_protocol/mesh_network.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "fixedpoint_tensor.h"
#include "core/privc/triplet_generator.h"
#include "core/privc3/paddle_tensor.h"

namespace privc {

using g_ctx_holder = paddle::mpc::ContextHolder;
using Fix64N32 = FixedPointTensor<int64_t, SCALING_N>;
using AbstractContext = paddle::mpc::AbstractContext;

class GCTest : public ::testing::Test {
public:

    static paddle::platform::CPUDeviceContext _cpu_ctx;
    static std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
    static std::shared_ptr<AbstractContext> _mpc_ctx[2];
    static std::shared_ptr<gloo::rendezvous::HashStore> _store;
    static std::thread _t[2];
    static std::shared_ptr<TensorAdapterFactory> _s_tensor_factory;

    virtual ~GCTest() noexcept {}

    static void SetUpTestCase() {

        paddle::framework::OperatorBase* op = nullptr;
        paddle::framework::Scope scope;
        paddle::framework::RuntimeContext ctx({}, {});

        _exec_ctx = std::make_shared<paddle::framework::ExecutionContext>(
            *op, scope, _cpu_ctx, ctx);
        _store = std::make_shared<gloo::rendezvous::HashStore>();

        for (size_t i = 0; i < 2; ++i) {
            _t[i] = std::thread(&GCTest::gen_mpc_ctx, i);
        }
        for (auto& ti : _t) {
            ti.join();
        }

        for (size_t i = 0; i < 2; ++i) {
            _t[i] = std::thread(&GCTest::init_ot_and_triplet, i);
        }
        for (auto& ti : _t) {
            ti.join();
        }
        _s_tensor_factory = std::make_shared<aby3::PaddleTensorFactory>(&_cpu_ctx);
    }

    static inline std::shared_ptr<paddle::mpc::MeshNetwork> gen_network(size_t idx) {
        return std::make_shared<paddle::mpc::MeshNetwork>(idx,
                                                          "127.0.0.1",
                                                          2,
                                                          "test_prefix_privc",
                                                          _store);
    }
    static inline void gen_mpc_ctx(size_t idx) {
        auto net = gen_network(idx);
        net->init();
        _mpc_ctx[idx] = std::make_shared<PrivCContext>(idx, net);
    }

    static inline void init_ot_and_triplet(size_t idx) {
        std::shared_ptr<OT> ot = std::make_shared<OT>(_mpc_ctx[idx]);
        ot->init();
        std::dynamic_pointer_cast<PrivCContext>(_mpc_ctx[idx])->set_ot(ot);

        std::shared_ptr<TripletGenerator<int64_t, SCALING_N>> tripletor
                    = std::make_shared<TripletGenerator<int64_t, SCALING_N>>(_mpc_ctx[idx]);
        std::dynamic_pointer_cast<PrivCContext>(_mpc_ctx[idx])->set_triplet_generator(tripletor);
    }

    std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
        return _s_tensor_factory->template create<int64_t>(shape);
    }
};

std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
    return g_ctx_holder::tensor_factory()->template create<int64_t>(shape);
}

void test_closure_gc(int party_flag, std::vector<double>& output) {

    Fix64gc<SCALING_N> af(7.0, 0);
    Fix64gc<SCALING_N> bf(3.0, 1);
    Fix64gc<SCALING_N> cf = af;
    Fix64gc<SCALING_N> df(std::move(cf));
    df = bf;

    output.emplace_back(af.geq(bf).reconstruct());
    output.emplace_back(af.equal(bf).reconstruct());
    Fix64gc<SCALING_N> res = af + bf;
    output.emplace_back(res.reconstruct());
    res = af - bf;
    output.emplace_back(res.reconstruct());
    res = -af;
    output.emplace_back(res.reconstruct());
    res = af * bf;
    output.emplace_back(res.reconstruct());
    res = af / bf;
    output.emplace_back(res.reconstruct());
    res = af ^ bf;
    output.emplace_back(res.reconstruct());
    res = af.abs();
    output.emplace_back(res.reconstruct());
    output.emplace_back(af[34].reconstruct());

}

TEST_F(GCTest, gc_closure) {

    std::vector<double> output[2];

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                test_closure_gc(0, output[0]);
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                test_closure_gc(1, output[1]);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    int i = 0;

    auto check_output_equal = [&] () -> bool {
        if (output[0].size() != output[1].size()) {
            return false;
        }
        for (unsigned int i = 0; i < output[0].size(); i += 1) {
            if (output[0][i] != output[1][i]) {
                return false;
            }
        }
        return true;
     };
    ASSERT_TRUE(check_output_equal());

    ASSERT_FLOAT_EQ(1, output[0][i++]);
    ASSERT_FLOAT_EQ(0, output[0][i++]);
    ASSERT_FLOAT_EQ(10, output[0][i++]);
    ASSERT_FLOAT_EQ(4, output[0][i++]);
    ASSERT_FLOAT_EQ(-7, output[0][i++]);
    ASSERT_FLOAT_EQ(21, output[0][i++]);
    ASSERT_FLOAT_EQ(7.0 / 3, output[0][i++]);
    ASSERT_FLOAT_EQ(4, output[0][i++]);
    ASSERT_FLOAT_EQ(7, output[0][i++]);
    ASSERT_FLOAT_EQ(1, output[0][i++]);
}

paddle::platform::CPUDeviceContext privc::GCTest::_cpu_ctx;
std::shared_ptr<paddle::framework::ExecutionContext> privc::GCTest::_exec_ctx;
std::shared_ptr<AbstractContext> privc::GCTest::_mpc_ctx[2];
std::shared_ptr<gloo::rendezvous::HashStore> privc::GCTest::_store;
std::thread privc::GCTest::_t[2];
std::shared_ptr<TensorAdapterFactory> privc::GCTest::_s_tensor_factory;

} // namespace privc
