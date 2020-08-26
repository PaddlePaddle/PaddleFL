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

#include <algorithm>
#include <memory>
#include <thread>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/platform/init.h"

#include "boolean_tensor.h"
#include "fixedpoint_tensor.h"
#include "paddle_tensor.h"
#include "aby3_context.h"
#include "core/paddlefl_mpc/mpc_protocol/mesh_network.h"

namespace aby3 {

using paddle::framework::Tensor;
using AbstractContext = paddle::mpc::AbstractContext;

class BooleanTensorTest : public ::testing::Test {
public:
    paddle::platform::CPUDeviceContext _cpu_ctx;

    std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
    std::shared_ptr<AbstractContext> _mpc_ctx[3];

    std::shared_ptr<gloo::rendezvous::HashStore> _store;

    std::thread _t[3];

    std::shared_ptr<TensorAdapterFactory> _tensor_factory;

    virtual ~BooleanTensorTest() noexcept {}

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
            _t[i] = std::thread(&BooleanTensorTest::gen_mpc_ctx, this, i);
        }

        for (auto& ti : _t) {
            ti.join();
        }

        _tensor_factory = std::make_shared<PaddleTensorFactory>(&_cpu_ctx);
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

    std::shared_ptr<TensorAdapter<int64_t>> gen1() {
        return _tensor_factory->template create<int64_t>(std::vector<size_t>{1});
    }

    std::shared_ptr<TensorAdapter<int64_t>> gen(const std::vector<size_t>& shape) {
        return _tensor_factory->template create<int64_t>(shape);
    }
};

using paddle::mpc::ContextHolder;

TEST_F(BooleanTensorTest, empty_test) {
    ContextHolder::template run_with_context(_exec_ctx.get(), _mpc_ctx[0], [](){ ; });
}

using BTensor = BooleanTensor<int64_t>;

TEST_F(BooleanTensorTest, reveal1_test) {
    std::shared_ptr<TensorAdapter<int64_t>> s[3] = { gen1(), gen1(), gen1() };
    auto p = gen1();
    s[0]->data()[0] = 2;
    s[1]->data()[0] = 3;
    s[2]->data()[0] = 4;

    BTensor b0(s[0].get(), s[1].get());
    BTensor b1(s[1].get(), s[2].get());
    BTensor b2(s[2].get(), s[0].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(2 ^ 3 ^ 4, p->data()[0]);
}

TEST_F(BooleanTensorTest, reveal2_test) {
    std::shared_ptr<TensorAdapter<int64_t>> s[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> p[3] = { gen1(), gen1(), gen1() };
    s[0]->data()[0] = 2;
    s[1]->data()[0] = 3;
    s[2]->data()[0] = 4;

    BTensor b0(s[0].get(), s[1].get());
    BTensor b1(s[1].get(), s[2].get());
    BTensor b2(s[2].get(), s[0].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0.reveal(p[0].get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1.reveal(p[1].get());
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2.reveal(p[2].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(2 ^ 3 ^ 4, p[0]->data()[0]);
    EXPECT_EQ(2 ^ 3 ^ 4, p[1]->data()[0]);
    EXPECT_EQ(2 ^ 3 ^ 4, p[2]->data()[0]);
}

TEST_F(BooleanTensorTest, xor1_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sr[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    // rhs = 7 = 1 ^ 2 ^ 4
    sr[0]->data()[0] = 1;
    sr[1]->data()[0] = 2;
    sr[2]->data()[0] = 4;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor br0(sr[0].get(), sr[1].get());
    BTensor br1(sr[1].get(), sr[2].get());
    BTensor br2(sr[2].get(), sr[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bitwise_xor(&br0, &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bitwise_xor(&br1, &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bitwise_xor(&br2, &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 ^ 7, p->data()[0]);
}

TEST_F(BooleanTensorTest, xor2_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };

    auto pr = gen1();

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    // rhs = 7
    pr->data()[0] = 7;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bitwise_xor(pr.get(),  &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bitwise_xor(pr.get(), &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bitwise_xor(pr.get(), &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 ^ 7, p->data()[0]);
}

TEST_F(BooleanTensorTest, and1_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sr[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    // rhs = 7 = 1 ^ 2 ^ 4
    sr[0]->data()[0] = 1;
    sr[1]->data()[0] = 2;
    sr[2]->data()[0] = 4;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor br0(sr[0].get(), sr[1].get());
    BTensor br1(sr[1].get(), sr[2].get());
    BTensor br2(sr[2].get(), sr[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bitwise_and(&br0, &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bitwise_and(&br1, &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bitwise_and(&br2, &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 & 7, p->data()[0]);
}

TEST_F(BooleanTensorTest, and2_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };

    auto pr = gen1();

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    // rhs = 7
    pr->data()[0] = 7;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bitwise_and(pr.get(),  &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bitwise_and(pr.get(), &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bitwise_and(pr.get(), &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 & 7, p->data()[0]);
}

TEST_F(BooleanTensorTest, or1_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sr[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    // rhs = 7 = 1 ^ 2 ^ 4
    sr[0]->data()[0] = 1;
    sr[1]->data()[0] = 2;
    sr[2]->data()[0] = 4;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor br0(sr[0].get(), sr[1].get());
    BTensor br1(sr[1].get(), sr[2].get());
    BTensor br2(sr[2].get(), sr[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bitwise_or(&br0, &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bitwise_or(&br1, &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bitwise_or(&br2, &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 | 7, p->data()[0]);
}

TEST_F(BooleanTensorTest, or2_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };

    auto pr = gen1();

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    // rhs = 7
    pr->data()[0] = 7;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bitwise_or(pr.get(),  &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bitwise_or(pr.get(), &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bitwise_or(pr.get(), &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 | 7, p->data()[0]);
}

TEST_F(BooleanTensorTest, not_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bitwise_not(&bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bitwise_not(&bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bitwise_not(&bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(~5, p->data()[0]);
}

TEST_F(BooleanTensorTest, lshift_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.lshift(1, &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.lshift(1, &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.lshift(1, &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 << 1, p->data()[0]);
}

TEST_F(BooleanTensorTest, rshift_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.rshift(1, &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.rshift(1, &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.rshift(1, &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 >> 1, p->data()[0]);
}

TEST_F(BooleanTensorTest, logical_rshift_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };

    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = -1
    sl[0]->data()[0] = -1;
    sl[1]->data()[0] = 0;
    sl[2]->data()[0] = 0;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.logical_rshift(1, &bout0);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.logical_rshift(1, &bout1);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.logical_rshift(1, &bout2);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(-1ull >> 1, p->data()[0]);
}

TEST_F(BooleanTensorTest, ppa_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sr[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1() };

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    // rhs = 7 = 1 ^ 2 ^ 4
    sr[0]->data()[0] = 1;
    sr[1]->data()[0] = 2;
    sr[2]->data()[0] = 4;

    auto p = gen1();

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor br0(sr[0].get(), sr[1].get());
    BTensor br1(sr[1].get(), sr[2].get());
    BTensor br2(sr[2].get(), sr[0].get());

    BTensor bout0(sout[0].get(), sout[1].get());
    BTensor bout1(sout[2].get(), sout[3].get());
    BTensor bout2(sout[4].get(), sout[5].get());

    const size_t nbits = 64;

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.ppa(&br0, &bout0, nbits);
                bout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.ppa(&br1, &bout1, nbits);
                bout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.ppa(&br2, &bout2, nbits);
                bout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5 + 7, p->data()[0]);
}

using FTensor = FixedPointTensor<int64_t, 32u>;

TEST_F(BooleanTensorTest, b2a_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1()};

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    BTensor b0(sl[0].get(), sl[1].get());
    BTensor b1(sl[1].get(), sl[2].get());
    BTensor b2(sl[2].get(), sl[0].get());

    FTensor f0(sout[0].get(), sout[1].get());
    FTensor f1(sout[2].get(), sout[3].get());
    FTensor f2(sout[4].get(), sout[5].get());

    auto p = gen1();

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0.b2a(&f0);
                f0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1.b2a(&f1);
                f1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2.b2a(&f2);
                f2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(5, p->data()[0]);
}

TEST_F(BooleanTensorTest, a2b_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1()};

    // lhs = 9 = 2 + 3 + 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    BTensor b0(sout[0].get(), sout[1].get());
    BTensor b1(sout[2].get(), sout[3].get());
    BTensor b2(sout[4].get(), sout[5].get());

    FTensor f0(sl[0].get(), sl[1].get());
    FTensor f1(sl[1].get(), sl[2].get());
    FTensor f2(sl[2].get(), sl[0].get());

    auto p = gen1();

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0 = &f0;
                b0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1 = &f1;
                b1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2 = &f2;
                b2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(9, p->data()[0]);
}

TEST_F(BooleanTensorTest, bit_extract_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1()};

    // lhs = 9 = 2 + 3 + 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    FTensor f0(sl[0].get(), sl[1].get());
    FTensor f1(sl[1].get(), sl[2].get());
    FTensor f2(sl[2].get(), sl[0].get());

    BTensor b0(sout[0].get(), sout[1].get());
    BTensor b1(sout[2].get(), sout[3].get());
    BTensor b2(sout[4].get(), sout[5].get());

    auto p = gen1();

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0.bit_extract(3, &f0);
                b0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1.bit_extract(3, &f1);
                b1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2.bit_extract(3, &f2);
                b2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(1, p->data()[0]);
}

TEST_F(BooleanTensorTest, boolean_bit_extract_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1()};

    // lhs = 5 = 2 ^ 3 ^ 4
    sl[0]->data()[0] = 2;
    sl[1]->data()[0] = 3;
    sl[2]->data()[0] = 4;

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    BTensor b0(sout[0].get(), sout[1].get());
    BTensor b1(sout[2].get(), sout[3].get());
    BTensor b2(sout[4].get(), sout[5].get());

    auto p = gen1();

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.bit_extract(2, &b0);
                b0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.bit_extract(2, &b1);
                b1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.bit_extract(2, &b2);
                b2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(1, p->data()[0]);
}

TEST_F(BooleanTensorTest, bit_extract_test2) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1()};

    // lhs = -9 = -2 + -3 + -4
    sl[0]->data()[0] = -2;
    sl[1]->data()[0] = -3;
    sl[2]->data()[0] = -4;

    FTensor f0(sl[0].get(), sl[1].get());
    FTensor f1(sl[1].get(), sl[2].get());
    FTensor f2(sl[2].get(), sl[0].get());

    BTensor b0(sout[0].get(), sout[1].get());
    BTensor b1(sout[2].get(), sout[3].get());
    BTensor b2(sout[4].get(), sout[5].get());

    auto p = gen1();

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0.bit_extract(63, &f0);
                b0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1.bit_extract(63, &f1);
                b1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2.bit_extract(63, &f2);
                b2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(1, p->data()[0]);
}

TEST_F(BooleanTensorTest, bit_extract_test3) {
    std::vector<size_t> shape = {2, 2};
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen(shape), gen(shape), gen(shape) };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen(shape), gen(shape), gen(shape),
                                                        gen(shape), gen(shape), gen(shape)};

    // lhs = -65536
    sl[0]->data()[0] = 626067816440182033;
    sl[1]->data()[0] = 1108923486657625775;
    sl[2]->data()[0] = -1734991303097873344;

    sl[0]->data()[1] = -1320209182212830031 ;
    sl[1]->data()[1] = 3175682926293206038;
    sl[2]->data()[1] = -1855473744080441543;

    sl[0]->data()[2] = -7241979567589308516;
    sl[1]->data()[2] = 5579083190137080035;
    sl[2]->data()[2] = 1662896377452162945;

    sl[0]->data()[3] = 1468124374943170272;
    sl[1]->data()[3] = -4796789375126030707;
    sl[2]->data()[3] = 3328665000182794899;

    FTensor f0(sl[0].get(), sl[1].get());
    FTensor f1(sl[1].get(), sl[2].get());
    FTensor f2(sl[2].get(), sl[0].get());

    BTensor b0(sout[0].get(), sout[1].get());
    BTensor b1(sout[2].get(), sout[3].get());
    BTensor b2(sout[4].get(), sout[5].get());

    auto p = gen(shape);

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0.bit_extract(63, &f0);
                b0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1.bit_extract(63, &f1);
                b1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2.bit_extract(63, &f2);
                b2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(1, p->data()[0]);
    EXPECT_EQ(1, p->data()[1]);
    EXPECT_EQ(1, p->data()[2]);
    EXPECT_EQ(1, p->data()[3]);
}

TEST_F(BooleanTensorTest, abmul_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1()};

    // lhs = 1
    sl[0]->data()[0] = 1;
    sl[1]->data()[0] = 0;
    sl[2]->data()[0] = 0;

    BTensor b0(sl[0].get(), sl[1].get());
    BTensor b1(sl[1].get(), sl[2].get());
    BTensor b2(sl[2].get(), sl[0].get());

    FTensor f0(sout[0].get(), sout[1].get());
    FTensor f1(sout[2].get(), sout[3].get());
    FTensor f2(sout[4].get(), sout[5].get());

    auto p = gen1();

    // rhs = 7
    p->data()[0] = 7;

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                b0.mul(p.get(), &f0, 0);
                f0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                b1.mul(nullptr, &f1, 0);
                f1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                b2.mul(nullptr, &f2, 0);
                f2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(1 * 7, p->data()[0]);
}

TEST_F(BooleanTensorTest, abmul2_test) {
    std::shared_ptr<TensorAdapter<int64_t>> sl[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sr[3] = { gen1(), gen1(), gen1() };
    std::shared_ptr<TensorAdapter<int64_t>> sout[6] = { gen1(), gen1(), gen1(),
                                                        gen1(), gen1(), gen1()};

    // lhs = 1
    sl[0]->data()[0] = 1;
    sl[1]->data()[0] = 0;
    sl[2]->data()[0] = 0;

    // rhs = 12 = 3 + 4 + 5
    sr[0]->data()[0] = 3;
    sr[1]->data()[0] = 4;
    sr[2]->data()[0] = 5;

    BTensor bl0(sl[0].get(), sl[1].get());
    BTensor bl1(sl[1].get(), sl[2].get());
    BTensor bl2(sl[2].get(), sl[0].get());

    FTensor fr0(sr[0].get(), sr[1].get());
    FTensor fr1(sr[1].get(), sr[2].get());
    FTensor fr2(sr[2].get(), sr[0].get());

    FTensor fout0(sout[0].get(), sout[1].get());
    FTensor fout1(sout[2].get(), sout[3].get());
    FTensor fout2(sout[4].get(), sout[5].get());

    auto p = gen1();

    _t[0] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                bl0.mul(&fr0, &fout0);
                fout0.reveal_to_one(0, p.get());
            });
        }
    );

    _t[1] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                bl1.mul(&fr1, &fout1);
                fout1.reveal_to_one(0, nullptr);
            });
        }
    );

    _t[2] = std::thread(
        [&] () {
        ContextHolder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[2], [&](){
                bl2.mul(&fr2, &fout2);
                fout2.reveal_to_one(0, nullptr);
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }
    EXPECT_EQ(1 * 12, p->data()[0]);
}
} // namespace aby3
