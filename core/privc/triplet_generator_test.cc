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

class TripletGeneratorTest : public ::testing::Test {
public:
    static paddle::platform::CPUDeviceContext _cpu_ctx;
    static std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
    static std::shared_ptr<AbstractContext> _mpc_ctx[2];
    static std::shared_ptr<gloo::rendezvous::HashStore> _store;
    static std::thread _t[2];
    static std::shared_ptr<TensorAdapterFactory> _s_tensor_factory;

    virtual ~TripletGeneratorTest() noexcept {}

    static void SetUpTestCase() {

        paddle::framework::OperatorBase* op = nullptr;
        paddle::framework::Scope scope;
        paddle::framework::RuntimeContext ctx({}, {});

        _exec_ctx = std::make_shared<paddle::framework::ExecutionContext>(
            *op, scope, _cpu_ctx, ctx);
        _store = std::make_shared<gloo::rendezvous::HashStore>();

        for (size_t i = 0; i < 2; ++i) {
            _t[i] = std::thread(&TripletGeneratorTest::gen_mpc_ctx, i);
        }
        for (auto& ti : _t) {
            ti.join();
        }

        for (size_t i = 0; i < 2; ++i) {
            _t[i] = std::thread(&TripletGeneratorTest::init_ot_and_triplet, i);
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


TEST_F(TripletGeneratorTest, triplet) {
    std::vector<size_t> shape = { 1 };

    auto shape_triplet = shape;
    shape_triplet.insert(shape_triplet.begin(), 3);

    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen(shape_triplet), gen(shape_triplet)};

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                std::dynamic_pointer_cast<PrivCContext>(_mpc_ctx[0])
                        ->triplet_generator()->get_triplet(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                std::dynamic_pointer_cast<PrivCContext>(_mpc_ctx[1])
                        ->triplet_generator()->get_triplet(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    auto num_triplet = ret[0]->numel() / 3;
    for (int i = 0; i < ret[0]->numel() / 3; ++i) {
        auto ret0_ptr = ret[0]->data();
        auto ret1_ptr = ret[1]->data();

        uint64_t a_idx = i;
        uint64_t b_idx = num_triplet + i;
        uint64_t c_idx = 2 * num_triplet + i;
        int64_t c = fixed64_mult<SCALING_N>(*(ret0_ptr + a_idx), *(ret0_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret0_ptr + a_idx), *(ret1_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret1_ptr + a_idx), *(ret0_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret1_ptr + a_idx), *(ret1_ptr + b_idx));

        EXPECT_NEAR(c , (*(ret0_ptr + c_idx) + *(ret1_ptr + c_idx)), std::pow(2, SCALING_N * 0.00001));
    }
}

TEST_F(TripletGeneratorTest, penta_triplet) {
    std::vector<size_t> shape = { 1 };

    auto shape_triplet = shape;
    shape_triplet.insert(shape_triplet.begin(), 5);

    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen(shape_triplet), gen(shape_triplet)};

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                std::dynamic_pointer_cast<PrivCContext>(_mpc_ctx[0])
                        ->triplet_generator()->get_penta_triplet(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                std::dynamic_pointer_cast<PrivCContext>(_mpc_ctx[1])
                        ->triplet_generator()->get_penta_triplet(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    auto num_triplet = ret[0]->numel() / 5;
    for (int i = 0; i < ret[0]->numel() / 5; ++i) {
        auto ret0_ptr = ret[0]->data();
        auto ret1_ptr = ret[1]->data();

        uint64_t a_idx = i;
        uint64_t alpha_idx = num_triplet + i;
        uint64_t b_idx = 2 * num_triplet + i;
        uint64_t c_idx = 3 * num_triplet + i;
        uint64_t alpha_c_idx = 4 * num_triplet + i;
        int64_t c = fixed64_mult<SCALING_N>(*(ret0_ptr + a_idx), *(ret0_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret0_ptr + a_idx), *(ret1_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret1_ptr + a_idx), *(ret0_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret1_ptr + a_idx), *(ret1_ptr + b_idx));
        int64_t alpha_c = fixed64_mult<SCALING_N>(*(ret0_ptr + alpha_idx), *(ret0_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret0_ptr + alpha_idx), *(ret1_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret1_ptr + alpha_idx), *(ret0_ptr + b_idx))
                    + fixed64_mult<SCALING_N>(*(ret1_ptr + alpha_idx), *(ret1_ptr + b_idx));

        // sometimes the difference big than 200
        EXPECT_NEAR(c , (*(ret0_ptr + c_idx) + *(ret1_ptr + c_idx)), std::pow(2, SCALING_N * 0.00001));
        EXPECT_NEAR(alpha_c , (*(ret0_ptr + alpha_c_idx) + *(ret1_ptr + alpha_c_idx)), std::pow(2, SCALING_N * 0.00001));
    }
}

paddle::platform::CPUDeviceContext privc::TripletGeneratorTest::_cpu_ctx;
std::shared_ptr<paddle::framework::ExecutionContext> privc::TripletGeneratorTest::_exec_ctx;
std::shared_ptr<AbstractContext> privc::TripletGeneratorTest::_mpc_ctx[2];
std::shared_ptr<gloo::rendezvous::HashStore> privc::TripletGeneratorTest::_store;
std::thread privc::TripletGeneratorTest::_t[2];
std::shared_ptr<TensorAdapterFactory> privc::TripletGeneratorTest::_s_tensor_factory;

} // namespace privc
