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
#include "core/common/paddle_tensor.h"
#include "gc_bit.h"

namespace privc {

using g_ctx_holder = paddle::mpc::ContextHolder;
using Fix64N32 = FixedPointTensor<int64_t, SCALING_N>;
using AbstractContext = paddle::mpc::AbstractContext;

class BitTest : public ::testing::Test {
public:

    static paddle::platform::CPUDeviceContext _cpu_ctx;
    static std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
    static std::shared_ptr<AbstractContext> _mpc_ctx[2];
    static std::shared_ptr<gloo::rendezvous::HashStore> _store;
    static std::thread _t[2];
    static std::shared_ptr<TensorAdapterFactory> _s_tensor_factory;

    virtual ~BitTest() noexcept {}

    static void SetUpTestCase() {

        paddle::framework::OperatorBase* op = nullptr;
        paddle::framework::Scope scope;
        paddle::framework::RuntimeContext ctx({}, {});

        _exec_ctx = std::make_shared<paddle::framework::ExecutionContext>(
            *op, scope, _cpu_ctx, ctx);
        _store = std::make_shared<gloo::rendezvous::HashStore>();

        for (size_t i = 0; i < 2; ++i) {
            _t[i] = std::thread(&BitTest::gen_mpc_ctx, i);
        }
        for (auto& ti : _t) {
            ti.join();
        }

        _s_tensor_factory = std::make_shared<common::PaddleTensorFactory>(&_cpu_ctx);
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

    template<typename T = int64_t>
    std::shared_ptr<TensorAdapter<T>> gen(std::vector<size_t> shape) {
        return _s_tensor_factory->template create<T>(shape);
    }
};

std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
    return g_ctx_holder::tensor_factory()->template create<int64_t>(shape);
}

TEST_F(BitTest, reconstruct) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<u8>> sl = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> sr = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> ret[4] = {gen<u8>(shape), gen<u8>(shape),
                                        gen<u8>(shape), gen<u8>(shape)};

    sl->data()[0] = (int64_t)1;
    sl->data()[1] = (int64_t)1;
    sl->data()[2] = (int64_t)0;
    sl->data()[3] = (int64_t)0;
    
    sr->data()[0] = (int64_t)1;
    sr->data()[1] = (int64_t)0;
    sr->data()[2] = (int64_t)1;
    sr->data()[3] = (int64_t)0;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                BitTensor op0(sl.get(), 0);
                BitTensor op1(sr.get(), 1);

                op0.reconstruct(ret[0].get());
                op1.reconstruct(ret[2].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                BitTensor op0(sl.get(), 0);
                BitTensor op1(sr.get(), 1);

                op0.reconstruct(ret[1].get());
                op1.reconstruct(ret[3].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    bool reconstruct_true = std::equal(ret[0]->data(),
                            ret[0]->data() + ret[0]->numel(),
                            ret[1]->data());
    bool reconstruct_true1 = std::equal(ret[2]->data(),
                            ret[2]->data() + ret[2]->numel(),
                            ret[3]->data());
    EXPECT_TRUE(reconstruct_true);
    EXPECT_TRUE(reconstruct_true1);

    EXPECT_EQ(1, ret[0]->data()[0]);
    EXPECT_EQ(1, ret[0]->data()[1]);
    EXPECT_EQ(0, ret[0]->data()[2]);
    EXPECT_EQ(0, ret[0]->data()[3]);

    EXPECT_EQ(1, ret[2]->data()[0]);
    EXPECT_EQ(0, ret[2]->data()[1]);
    EXPECT_EQ(1, ret[2]->data()[2]);
    EXPECT_EQ(0, ret[2]->data()[3]);
}

TEST_F(BitTest, xor) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<u8>> sl = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> sr = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> ret[2] = {gen<u8>(shape), gen<u8>(shape)};

    sl->data()[0] = (int64_t)1;
    sl->data()[1] = (int64_t)1;
    sl->data()[2] = (int64_t)0;
    sl->data()[3] = (int64_t)0;
    
    sr->data()[0] = (int64_t)1;
    sr->data()[1] = (int64_t)0;
    sr->data()[2] = (int64_t)1;
    sr->data()[3] = (int64_t)0;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                BitTensor lhs(sl.get(), 0);
                BitTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_xor(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                BitTensor lhs(sl.get(), 0);
                BitTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_xor(&rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(0, ret[0]->data()[0]);
    EXPECT_EQ(1, ret[0]->data()[1]);
    EXPECT_EQ(1, ret[0]->data()[2]);
    EXPECT_EQ(0, ret[0]->data()[3]);
}

TEST_F(BitTest, and) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<u8>> sl = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> sr = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> ret[2] = {gen<u8>(shape), gen<u8>(shape)};

    sl->data()[0] = (int64_t)1;
    sl->data()[1] = (int64_t)1;
    sl->data()[2] = (int64_t)0;
    sl->data()[3] = (int64_t)0;
    
    sr->data()[0] = (int64_t)1;
    sr->data()[1] = (int64_t)0;
    sr->data()[2] = (int64_t)1;
    sr->data()[3] = (int64_t)0;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                BitTensor lhs(sl.get(), 0);
                BitTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_and(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                BitTensor lhs(sl.get(), 0);
                BitTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_and(&rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(1, ret[0]->data()[0]);
    EXPECT_EQ(0, ret[0]->data()[1]);
    EXPECT_EQ(0, ret[0]->data()[2]);
    EXPECT_EQ(0, ret[0]->data()[3]);
}

TEST_F(BitTest, or) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<u8>> sl = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> sr = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> ret[2] = {gen<u8>(shape), gen<u8>(shape)};

    sl->data()[0] = (int64_t)1;
    sl->data()[1] = (int64_t)1;
    sl->data()[2] = (int64_t)0;
    sl->data()[3] = (int64_t)0;
    
    sr->data()[0] = (int64_t)1;
    sr->data()[1] = (int64_t)0;
    sr->data()[2] = (int64_t)1;
    sr->data()[3] = (int64_t)0;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                BitTensor lhs(sl.get(), 0);
                BitTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_or(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                BitTensor lhs(sl.get(), 0);
                BitTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_or(&rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(1, ret[0]->data()[0]);
    EXPECT_EQ(1, ret[0]->data()[1]);
    EXPECT_EQ(1, ret[0]->data()[2]);
    EXPECT_EQ(0, ret[0]->data()[3]);
}

TEST_F(BitTest, not) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<u8>> sl = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<u8>> ret[2] = {gen<u8>(shape), gen<u8>(shape)};

    sl->data()[0] = (int64_t)1;
    sl->data()[1] = (int64_t)0;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)0;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                BitTensor lhs(sl.get(), 0);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_not(&ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                BitTensor lhs(sl.get(), 0);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.bitwise_not(&ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(0, ret[0]->data()[0]);
    EXPECT_EQ(1, ret[0]->data()[1]);
    EXPECT_EQ(0, ret[0]->data()[2]);
    EXPECT_EQ(1, ret[0]->data()[3]);
}


paddle::platform::CPUDeviceContext privc::BitTest::_cpu_ctx;
std::shared_ptr<paddle::framework::ExecutionContext> privc::BitTest::_exec_ctx;
std::shared_ptr<AbstractContext> privc::BitTest::_mpc_ctx[2];
std::shared_ptr<gloo::rendezvous::HashStore> privc::BitTest::_store;
std::thread privc::BitTest::_t[2];
std::shared_ptr<TensorAdapterFactory> privc::BitTest::_s_tensor_factory;

} // namespace privc
