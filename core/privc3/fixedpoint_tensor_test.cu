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

#include "gtest/gtest.h"

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

#include "core/paddlefl_mpc/mpc_protocol/network/cuda_copy_network.h"
#include "core/paddlefl_mpc/mpc_protocol/context_holder.h"
#include "core/paddlefl_mpc/mpc_protocol/abstract_context.h"
#include "core/common/paddle_tensor_impl.cu.h"

#include "aby3_context.h"
#include "fixedpoint_tensor.h"
#include "fixedpoint_tensor_imp.h"

namespace aby3 {

using g_ctx_holder = paddle::mpc::ContextHolder;
template<typename T>
using CudaPaddleTensor = common::CudaPaddleTensor<T>;

using Fix64N16 = FixedPointTensor<int64_t, 16>;
using BTensor = BooleanTensor<int64_t>;
using AbstractContext = paddle::mpc::AbstractContext;

class FixedTensorTest : public ::testing::Test {
public:

    const paddle::platform::CUDADeviceContext* get_gpu_ctx() {
        paddle::platform::CUDAPlace gpu(0);
        auto& pool = paddle::platform::DeviceContextPool::Instance();
        return pool.template GetByPlace<paddle::platform::CUDAPlace>(gpu);
    }

    std::shared_ptr<paddle::framework::ExecutionContext> _exec_ctx;
    std::shared_ptr<AbstractContext> _mpc_ctx[3];
    std::shared_ptr<TensorAdapterFactory> _s_tensor_factory;

    std::thread _t[3];

    static void SetUpTestCase() {
         paddle::platform::CUDAPlace gpu(0);
         paddle::platform::DeviceContextPool::Init({gpu});

         auto& pool = paddle::platform::DeviceContextPool::Instance();
         paddle::mpc::AbstractContext::_s_stream = pool.GetByPlace(gpu)->stream();
    }

    void SetUp() {
         paddle::framework::OperatorBase* op = nullptr;
         paddle::framework::Scope scope;
         paddle::framework::RuntimeContext ctx({}, {});
         // only device_ctx is needed

         _exec_ctx = std::make_shared<paddle::framework::ExecutionContext>(
             *op, scope, *get_gpu_ctx(), ctx);

         _s_tensor_factory = std::make_shared<::common::CudaPaddleTensorFactory>(get_gpu_ctx());

         std::thread t[3];

         for (size_t i = 0; i < 3; ++i) {
             _t[i] = std::thread(&FixedTensorTest::gen_mpc_ctx, this, i);
             // using namespace std::chrono_literals;
             // std::this_thread::sleep_for(20ms);
         }
         for (auto& ti : _t) {
             ti.join();
         }

    }

    void gen_mpc_ctx(size_t idx) {
        auto net = std::make_shared<paddle::mpc::CudaCopyNetwork>(idx, 3,
                    get_gpu_ctx()->stream());
        // net->init();
        _mpc_ctx[idx] = std::make_shared<ABY3Context>(idx, net);
    }

    std::shared_ptr<TensorAdapter<int64_t> > gen(float val, std::vector<size_t> shape) {
        auto ret = _s_tensor_factory->template create<int64_t>(shape);
        dynamic_cast<CudaPaddleTensor<int64_t>*>(ret.get())->from_float_point_scalar(val, shape, 16);
        return ret;
    }
};

using paddle::mpc::ContextHolder;
#define TEST_SIZE 2
#define COMMA ,
#define TEST_METHOD_CIPHER(method, buf) __TEST_METHOD_CIPHER(method, (&rhs) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, Fix64N16)

#define TEST_METHOD_CIPHER_PLAIN(method, buf) __TEST_METHOD_CIPHER(method, (tensor[0].get()) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, Fix64N16)

#define TEST_METHOD_MONO(method, buf) __TEST_METHOD_CIPHER(method, , buf, TEST_SIZE COMMA TEST_SIZE, Fix64N16)
#define __TEST_METHOD_CIPHER(method, input, buf, ret_shape, ret_type) do {\
    std::shared_ptr<TensorAdapter<int64_t> > tensor[9]; \
    for (int i = 0; i < 2; i++) { \
        tensor[i] = gen(1.23, {TEST_SIZE, TEST_SIZE}); \
    } \
    for (int i = 2; i < 9; i++) { \
        tensor[i] = gen(1.23, { ret_shape }); \
    } \
    Fix64N16 lhs(tensor[0].get(), tensor[1].get()); \
    Fix64N16 rhs(tensor[0].get(), tensor[1].get()); \
    ret_type ret0(tensor[2].get(), tensor[3].get()); \
    ret_type ret1(tensor[4].get(), tensor[5].get()); \
    ret_type ret2(tensor[6].get(), tensor[7].get()); \
    _t[0] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[0], \
            [&](){ \
            lhs.method(input &ret0); \
            ret0.reveal_to_one(0, tensor[8].get()); \
            }); \
        } \
    ); \
    _t[1] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[1], \
            [&](){ \
            lhs.method(input &ret1); \
            ret1.reveal_to_one(0, nullptr); \
            }); \
        } \
    ); \
    _t[2] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[2], \
            [&](){ \
            lhs.method(input &ret2); \
            ret2.reveal_to_one(0, nullptr); \
            }); \
        } \
    ); \
    for (auto &t: _t) { \
        t.join(); \
    } \
    cudaMemcpy(buf, tensor[8]->data(), sizeof(buf), cudaMemcpyDeviceToHost); \
    cudaStreamSynchronize(get_gpu_ctx()->stream()); \
    } while(0)

TEST_F(FixedTensorTest, add) {
    int64_t buf[4];
    TEST_METHOD_CIPHER(add, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3 * 2, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, add_plain) {
    int64_t buf[4];

    TEST_METHOD_CIPHER_PLAIN(add, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 4, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, sub) {
    int64_t buf[4];

    TEST_METHOD_CIPHER(sub, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, sub_plain) {
    int64_t buf[4];

    TEST_METHOD_CIPHER_PLAIN(sub, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 2, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, mul) {
    int64_t buf[4];

    TEST_METHOD_CIPHER(mul, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3 * 1.23 * 3, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, mul_plain) {
    int64_t buf[4];

    TEST_METHOD_CIPHER_PLAIN(mul, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3 * 1.23, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, mat_mul) {
    int64_t buf[4];

    TEST_METHOD_CIPHER(mat_mul, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3 * 1.23 * 3 * TEST_SIZE, buf[i] / 65536.0, 0.1);
    }

}

TEST_F(FixedTensorTest, mat_mul_plain) {
    int64_t buf[4];

    TEST_METHOD_CIPHER_PLAIN(mat_mul, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3 * 1.23 * TEST_SIZE, buf[i] / 65536.0, 0.05);
    }

}

TEST_F(FixedTensorTest, neg) {
    int64_t buf[4];

    TEST_METHOD_MONO(negative, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(-1.23 * 3, buf[i] / 65536.0, 0.001);
    }

}

TEST_F(FixedTensorTest, div_plain) {
    int64_t buf[4];

    TEST_METHOD_CIPHER_PLAIN(div, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(3, buf[i] / 65536.0, 0.001);
    }

}

TEST_F(FixedTensorTest, div) {
    int64_t buf[4];

    TEST_METHOD_CIPHER(div, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i] / 65536.0, 0.001);
    }

}

TEST_F(FixedTensorTest, long_div) {
    int64_t buf[4];

    TEST_METHOD_CIPHER(long_div, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i] / 65536.0, 0.01);
    }

}

TEST_F(FixedTensorTest, inverse_square_root) {
    int64_t buf[4];

    TEST_METHOD_MONO(inverse_square_root, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0.52, buf[i] / 65536.0, 0.001);
    }

}

TEST_F(FixedTensorTest, exp) {
    int64_t buf[4];

    TEST_METHOD_MONO(exp, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(40, buf[i] / 65536.0, 2);
    }

}

TEST_F(FixedTensorTest, sum) {
    int64_t buf[1];

    __TEST_METHOD_CIPHER(sum, , buf, 1, Fix64N16);

    for (int i = 0; i < 1; ++i) {
        EXPECT_NEAR(3 * 1.23 * TEST_SIZE * TEST_SIZE, buf[i] / 65536.0, 0.001);
    }

}

TEST_F(FixedTensorTest, dot_mul) {
    int64_t buf[1];

    __TEST_METHOD_CIPHER(dot_mul, (&rhs) COMMA, buf, 1, Fix64N16);

    for (int i = 0; i < 1; ++i) {
        EXPECT_NEAR(3 *3 * 1.23 * 1.23 * TEST_SIZE * TEST_SIZE, buf[i] / 65536.0, 0.001);
    }

}

TEST_F(FixedTensorTest, dot_mul_plain) {
    int64_t buf[1];

    __TEST_METHOD_CIPHER(dot_mul, (tensor[0].get()) COMMA, buf, 1, Fix64N16);

    for (int i = 0; i < 1; ++i) {
        EXPECT_NEAR(3 * 1.23 * 1.23 * TEST_SIZE * TEST_SIZE, buf[i] / 65536.0, 0.001);
    }

}

TEST_F(FixedTensorTest, relu) {
    int64_t buf[4];

    TEST_METHOD_MONO(relu, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3, buf[i] / 65536.0, 2);
    }

}

TEST_F(FixedTensorTest, sigmoid) {
    int64_t buf[4];

    TEST_METHOD_MONO(sigmoid, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.0, buf[i] / 65536.0, 2);
    }

}

TEST_F(FixedTensorTest, sigmoid_enhanced) {
    int64_t buf[4];

    TEST_METHOD_MONO(sigmoid_enhanced, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.0, buf[i] / 65536.0, 2);
    }

}

TEST_F(FixedTensorTest, sigmoid_chebyshev) {
    int64_t buf[4];

    TEST_METHOD_MONO(sigmoid_chebyshev, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.0, buf[i] / 65536.0, 2);
    }

}

TEST_F(FixedTensorTest, sigmoid_high_precision) {
    int64_t buf[4];

    TEST_METHOD_MONO(sigmoid_high_precision, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.0, buf[i] / 65536.0, 2);
    }

}

TEST_F(FixedTensorTest, softmax) {
    int64_t buf[4];

    TEST_METHOD_MONO(softmax, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0.5, buf[i] / 65536.0, 2);
    }

}

TEST_F(FixedTensorTest, lt) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(lt, (&rhs) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, lt_plain) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(lt, (tensor[0].get()) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, leq) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(leq, (&rhs) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, leq_plain) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(leq, (tensor[0].get()) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, gt) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(gt, (&rhs) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, gt_plain) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(gt, (tensor[0].get()) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, geq) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(geq, (&rhs) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, geq_plain) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(geq, (tensor[0].get()) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, eq) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(eq, (&rhs) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, eq_plain) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(eq, (tensor[0].get()) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, neq) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(neq, (&rhs) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(0, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, neq_plain) {
    int64_t buf[4];

    __TEST_METHOD_CIPHER(neq, (tensor[0].get()) COMMA, buf, TEST_SIZE COMMA TEST_SIZE, BTensor);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i], 0.001);
    }

}

TEST_F(FixedTensorTest, max) {
    int64_t buf[4];
    TEST_METHOD_CIPHER(max, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, max_plain) {
    int64_t buf[4];
    TEST_METHOD_CIPHER_PLAIN(max, buf);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, max_pooling) {
    int64_t buf[4];
    __TEST_METHOD_CIPHER(max_pooling, , buf, 1 COMMA TEST_SIZE, Fix64N16);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1.23 * 3, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, preds_to_idx) {
    int64_t buf[4];
    std::shared_ptr<TensorAdapter<int64_t> > tensor[9]; \
    for (int i = 0; i < 9; i++) { \
        tensor[i] = gen(0.33, { 4 }); \
    } \
    Fix64N16 lhs(tensor[0].get(), tensor[1].get()); \
    Fix64N16 ret0(tensor[2].get(), tensor[3].get()); \
    Fix64N16 ret1(tensor[4].get(), tensor[5].get()); \
    Fix64N16 ret2(tensor[6].get(), tensor[7].get()); \
    _t[0] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[0], \
            [&](){ \
            Fix64N16::preds_to_indices(&lhs, &ret0); \
            ret0.reveal_to_one(0, tensor[8].get()); \
            }); \
        } \
    ); \
    _t[1] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[1], \
            [&](){ \
            Fix64N16::preds_to_indices(&lhs, &ret1); \
            ret1.reveal_to_one(0, nullptr); \
            }); \
        } \
    ); \
    _t[2] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[2], \
            [&](){ \
            Fix64N16::preds_to_indices(&lhs, &ret2); \
            ret2.reveal_to_one(0, nullptr); \
            }); \
        } \
    ); \
    for (auto &t: _t) { \
        t.join(); \
    } \
    cudaMemcpy(buf, tensor[8]->data(), sizeof(buf), cudaMemcpyDeviceToHost); \
    cudaStreamSynchronize(get_gpu_ctx()->stream()); \

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(1, buf[i] / 65536.0, 0.001);
    }
}

TEST_F(FixedTensorTest, tp_fp_fn) {
    int64_t buf[3];
    std::shared_ptr<TensorAdapter<int64_t> > tensor[9]; \
    for (int i = 0; i < 9; i++) { \
        tensor[i] = gen(0.33, { 3 }); \
    } \
    Fix64N16 lhs(tensor[0].get(), tensor[1].get()); \
    Fix64N16 ret0(tensor[2].get(), tensor[3].get()); \
    Fix64N16 ret1(tensor[4].get(), tensor[5].get()); \
    Fix64N16 ret2(tensor[6].get(), tensor[7].get()); \
    _t[0] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[0], \
            [&](){ \
            Fix64N16::calc_tp_fp_fn(&lhs, &lhs, &ret0); \
            ret0.reveal_to_one(0, tensor[8].get()); \
            }); \
        } \
    ); \
    _t[1] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[1], \
            [&](){ \
            Fix64N16::calc_tp_fp_fn(&lhs, &lhs, &ret1); \
            ret1.reveal_to_one(0, nullptr); \
            }); \
        } \
    ); \
    _t[2] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[2], \
            [&](){ \
            Fix64N16::calc_tp_fp_fn(&lhs, &lhs, &ret2); \
            ret2.reveal_to_one(0, nullptr); \
            }); \
        } \
    ); \
    for (auto &t: _t) { \
        t.join(); \
    } \
    cudaMemcpy(buf, tensor[8]->data(), sizeof(buf), cudaMemcpyDeviceToHost); \
    cudaStreamSynchronize(get_gpu_ctx()->stream()); \

    EXPECT_NEAR(3, buf[0] / 65536.0, 0.1);
    EXPECT_NEAR(0, buf[1] / 65536.0, 0.1);
    EXPECT_NEAR(0, buf[1] / 65536.0, 0.1);
}

TEST_F(FixedTensorTest, precision_recall) {
    int64_t buf[3];
    std::shared_ptr<TensorAdapter<int64_t> > tensor[9]; \
    for (int i = 0; i < 5; i++) { \
        tensor[i] = gen(3.33, { 3 }); \
    } \
    Fix64N16 lhs(tensor[0].get(), tensor[1].get()); \
    _t[0] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[0], \
            [&](){ \
            Fix64N16::calc_precision_recall(&lhs, tensor[2].get()); \
            }); \
        } \
    ); \
    _t[1] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[1], \
            [&](){ \
            Fix64N16::calc_precision_recall(&lhs, tensor[3].get()); \
            }); \
        } \
    ); \
    _t[2] = std::thread( \
        [&] () { \
        ContextHolder::template run_with_context( \
            _exec_ctx.get(), _mpc_ctx[2], \
            [&](){ \
            Fix64N16::calc_precision_recall(&lhs, tensor[4].get()); \
            }); \
        } \
    ); \
    for (auto &t: _t) { \
        t.join(); \
    } \
    cudaMemcpy(buf, tensor[2]->data(), sizeof(buf), cudaMemcpyDeviceToHost); \
    cudaStreamSynchronize(get_gpu_ctx()->stream()); \

    EXPECT_NEAR(0.5, buf[0] / 65536.0, 0.1);
    EXPECT_NEAR(0.5, buf[1] / 65536.0, 0.1);
    EXPECT_NEAR(0.5, buf[2] / 65536.0, 0.1);
}
} // namespace aby3
