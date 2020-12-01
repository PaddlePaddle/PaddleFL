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
#include "integer.h"

namespace privc {

using g_ctx_holder = paddle::mpc::ContextHolder;
using Fix64N32 = FixedPointTensor<int64_t, SCALING_N>;
using AbstractContext = paddle::mpc::AbstractContext;

class IntegerTest : public ::testing::Test {
public:

    static paddle::platform::CPUDeviceContext _cpu_ctx;
    static std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
    static std::shared_ptr<AbstractContext> _mpc_ctx[2];
    static std::shared_ptr<gloo::rendezvous::HashStore> _store;
    static std::thread _t[2];
    static std::shared_ptr<TensorAdapterFactory> _s_tensor_factory;

    virtual ~IntegerTest() noexcept {}

    static void SetUpTestCase() {

        paddle::framework::OperatorBase* op = nullptr;
        paddle::framework::Scope scope;
        paddle::framework::RuntimeContext ctx({}, {});

        _exec_ctx = std::make_shared<paddle::framework::ExecutionContext>(
            *op, scope, _cpu_ctx, ctx);
        _store = std::make_shared<gloo::rendezvous::HashStore>();

        for (size_t i = 0; i < 2; ++i) {
            _t[i] = std::thread(&IntegerTest::gen_mpc_ctx, i);
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
   /* void fun_slice(TensorAdapter<int64_t>* in) {
    std::vector<size_t> shape = { 1, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> s_in = gen(shape);
    in->slice(0, 1, s_in.get());
    *(s_in->data()) = 1;
    *(s_in->data() + 1) = 2;
}*/
};

std::shared_ptr<TensorAdapter<int64_t>> gen(std::vector<size_t> shape) {
    return g_ctx_holder::tensor_factory()->template create<int64_t>(shape);
}

/*TEST_F(IntegerTest, test_slice) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    sl->data()[0] = (int64_t)0;
    sl->data()[1] = (int64_t)1;
    sl->data()[2] = (int64_t)2;
    sl->data()[3] = -((int64_t)3);

    fun_slice(sl.get());

    EXPECT_EQ(1, sl->data()[0]);
    EXPECT_EQ(2, sl->data()[1]);
    EXPECT_EQ(2, sl->data()[2]);
    EXPECT_EQ(-3, sl->data()[3]);
}
*/


TEST_F(IntegerTest, reconstruct) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[4] = {gen<int64_t>(shape), gen<int64_t>(shape),
                                                      gen<int64_t>(shape), gen<int64_t>(shape)};

    sl->data()[0] = (int64_t)3;
    sl->data()[1] = (int64_t)-3;
    sl->data()[2] = (int64_t)1 << 20;
    sl->data()[3] = -((int64_t)1 << 20);
    
    sr->data()[0] = (int64_t)3;
    sr->data()[1] = (int64_t)-3;
    sr->data()[2] = (int64_t)1 << 20;
    sr->data()[3] = -((int64_t)1 << 20);

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor op0(sl.get(), 0);
                IntegerTensor op1(sr.get(), 1);

                op0.reconstruct(ret[0].get());
                op1.reconstruct(ret[2].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor op0(sl.get(), 0);
                IntegerTensor op1(sr.get(), 1);

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

    EXPECT_EQ(3, ret[0]->data()[0]);
    EXPECT_EQ(-3, ret[0]->data()[1]);
    EXPECT_EQ((int64_t)1 << 20, ret[0]->data()[2]);
    EXPECT_EQ(-((int64_t)1 << 20), ret[0]->data()[3]);

    EXPECT_EQ(3, ret[2]->data()[0]);
    EXPECT_EQ(-3, ret[2]->data()[1]);
    EXPECT_EQ((int64_t)1 << 20, ret[2]->data()[2]);
    EXPECT_EQ(-((int64_t)1 << 20), ret[2]->data()[3]);
}

TEST_F(IntegerTest, add) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen<int64_t>(shape), gen<int64_t>(shape)};

    sl->data()[0] = (int64_t)-1;
    sl->data()[1] = (int64_t)-2;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)2;
    
    sr->data()[0] = (int64_t)2;
    sr->data()[1] = (int64_t)1;
    sr->data()[2] = (int64_t)-2;
    sr->data()[3] = (int64_t)-1;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_add(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_add(&rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(1, ret[0]->data()[0]);
    EXPECT_EQ(-1, ret[0]->data()[1]);
    EXPECT_EQ(-1, ret[0]->data()[2]);
    EXPECT_EQ(1, ret[0]->data()[3]);
}

TEST_F(IntegerTest, sub) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen<int64_t>(shape), gen<int64_t>(shape)};

    sl->data()[0] = (int64_t)-1;
    sl->data()[1] = (int64_t)-2;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)2;
    
    sr->data()[0] = (int64_t)2;
    sr->data()[1] = (int64_t)1;
    sr->data()[2] = (int64_t)-2;
    sr->data()[3] = (int64_t)-1;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_sub(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_sub(&rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(-3, ret[0]->data()[0]);
    EXPECT_EQ(-3, ret[0]->data()[1]);
    EXPECT_EQ(3, ret[0]->data()[2]);
    EXPECT_EQ(3, ret[0]->data()[3]);
}

TEST_F(IntegerTest, mul) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen<int64_t>(shape), gen<int64_t>(shape)};

    sl->data()[0] = (int64_t)-1;
    sl->data()[1] = (int64_t)2;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)2;
    
    sr->data()[0] = (int64_t)2;
    sr->data()[1] = (int64_t)1;
    sr->data()[2] = (int64_t)2;
    sr->data()[3] = (int64_t)-1;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_mul(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_mul(&rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(-2, ret[0]->data()[0]);
    EXPECT_EQ(2, ret[0]->data()[1]);
    EXPECT_EQ(2, ret[0]->data()[2]);
    EXPECT_EQ(-2, ret[0]->data()[3]);
}

TEST_F(IntegerTest, neg) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen<int64_t>(shape), gen<int64_t>(shape)};

    sl->data()[0] = (int64_t)1;
    sl->data()[1] = (int64_t)-1;
    sl->data()[2] = (int64_t)0;
    sl->data()[3] = (int64_t)0;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_neg(&ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_neg(&ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(-1, ret[0]->data()[0]);
    EXPECT_EQ(1, ret[0]->data()[1]);
    EXPECT_EQ(0, ret[0]->data()[2]);
    EXPECT_EQ(0, ret[0]->data()[3]);
}

TEST_F(IntegerTest, xor) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen<int64_t>(shape), gen<int64_t>(shape)};

    sl->data()[0] = (int64_t)-1;
    sl->data()[1] = (int64_t)-2;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)2;
    
    sr->data()[0] = (int64_t)2;
    sr->data()[1] = (int64_t)1;
    sr->data()[2] = (int64_t)-2;
    sr->data()[3] = (int64_t)-1;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_xor(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.bitwise_xor(&rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(((int64_t) -1) ^ ((int64_t) 2), ret[0]->data()[0]);
    EXPECT_EQ(((int64_t) -2) ^ ((int64_t) 1), ret[0]->data()[1]);
    EXPECT_EQ(((int64_t) 1) ^ ((int64_t) -2), ret[0]->data()[2]);
    EXPECT_EQ(((int64_t) 2) ^ ((int64_t) -1), ret[0]->data()[3]);
}

TEST_F(IntegerTest, abs) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen<int64_t>(shape), gen<int64_t>(shape)};

    sl->data()[0] = (int64_t)1;
    sl->data()[1] = (int64_t)-1;
    sl->data()[2] = (int64_t)0;
    sl->data()[3] = (int64_t)0;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.abs(&ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                lhs.abs(&ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(1, ret[0]->data()[0]);
    EXPECT_EQ(1, ret[0]->data()[1]);
    EXPECT_EQ(0, ret[0]->data()[2]);
    EXPECT_EQ(0, ret[0]->data()[3]);
}

TEST_F(IntegerTest, equal) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<u8>> ret[2] = {gen<u8>(shape), gen<u8>(shape)};

    sl->data()[0] = (int64_t)-1;
    sl->data()[1] = (int64_t)-2;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)2;
    
    sr->data()[0] = (int64_t)-1;
    sr->data()[1] = (int64_t)-1;
    sr->data()[2] = (int64_t)2;
    sr->data()[3] = (int64_t)2;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.equal(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.equal(&rhs, &ret_);
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
    EXPECT_EQ(1, ret[0]->data()[3]);
}

TEST_F(IntegerTest, geq) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<u8>> ret[2] = {gen<u8>(shape), gen<u8>(shape)};

    sl->data()[0] = (int64_t)0;
    sl->data()[1] = (int64_t)-2;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)2;
    
    sr->data()[0] = (int64_t)-1;
    sr->data()[1] = (int64_t)-1;
    sr->data()[2] = (int64_t)2;
    sr->data()[3] = (int64_t)2;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.geq(&rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                BitTensor ret_(get_block_shape(sl->shape()));

                lhs.geq(&rhs, &ret_);
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
    EXPECT_EQ(1, ret[0]->data()[3]);
}

TEST_F(IntegerTest, if_then_else) {
    std::vector<size_t> shape = { 2, 2 };
    std::shared_ptr<TensorAdapter<u8>> ifv = gen<u8>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sl = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> sr = gen<int64_t>(shape);
    std::shared_ptr<TensorAdapter<int64_t>> ret[2] = {gen<int64_t>(shape), gen<int64_t>(shape)};

    ifv->data()[0] = (int64_t)0;
    ifv->data()[1] = (int64_t)1;
    ifv->data()[2] = (int64_t)1;
    ifv->data()[3] = (int64_t)0;

    sl->data()[0] = (int64_t)0;
    sl->data()[1] = (int64_t)-2;
    sl->data()[2] = (int64_t)1;
    sl->data()[3] = (int64_t)2;
    
    sr->data()[0] = (int64_t)-1;
    sr->data()[1] = (int64_t)-1;
    sr->data()[2] = (int64_t)2;
    sr->data()[3] = (int64_t)2;

    _t[0] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[0], [&](){
                BitTensor if_val(ifv.get(), 0);
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                IntegerTensor::if_then_else(&if_val, &lhs, &rhs, &ret_);
                ret_.reconstruct(ret[0].get());
            });
        }
    );
    _t[1] = std::thread(
        [&] () {
        g_ctx_holder::template run_with_context(
            _exec_ctx.get(), _mpc_ctx[1], [&](){
                BitTensor if_val(ifv.get(), 0);
                IntegerTensor lhs(sl.get(), 0);
                IntegerTensor rhs(sr.get(), 1);

                IntegerTensor ret_(get_gc_shape(sl->shape()));

                IntegerTensor::if_then_else(&if_val, &lhs, &rhs, &ret_);
                ret_.reconstruct(ret[1].get());
            });
        }
    );
    for (auto &t: _t) {
        t.join();
    }

    EXPECT_EQ(-1, ret[0]->data()[0]);
    EXPECT_EQ(-2, ret[0]->data()[1]);
    EXPECT_EQ(1, ret[0]->data()[2]);
    EXPECT_EQ(2, ret[0]->data()[3]);
}

paddle::platform::CPUDeviceContext privc::IntegerTest::_cpu_ctx;
std::shared_ptr<paddle::framework::ExecutionContext> privc::IntegerTest::_exec_ctx;
std::shared_ptr<AbstractContext> privc::IntegerTest::_mpc_ctx[2];
std::shared_ptr<gloo::rendezvous::HashStore> privc::IntegerTest::_store;
std::thread privc::IntegerTest::_t[2];
std::shared_ptr<TensorAdapterFactory> privc::IntegerTest::_s_tensor_factory;

} // namespace privc
