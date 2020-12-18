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

#include "paddle_tensor.h"

#include <algorithm>

#include "gtest/gtest.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/framework/tensor.h"

namespace common {

using paddle::platform::CPUDeviceContext;
using paddle::framework::Tensor;

class PaddleTensorTest : public ::testing::Test {
public:

    std::shared_ptr<TensorAdapterFactory> _tensor_factory;

    CPUDeviceContext _cpu_ctx;
    const static size_t SCALING_FACTOR = 32;

    virtual ~PaddleTensorTest() noexcept {}

    void SetUp() {
        _tensor_factory = std::make_shared<PaddleTensorFactory>(&_cpu_ctx);
    }
};

TEST_F(PaddleTensorTest, factory_test) {
    EXPECT_NO_THROW(_tensor_factory->template create<int64_t>());
    std::vector<size_t> shape = { 2, 3 };
    EXPECT_NO_THROW(_tensor_factory->template create<int64_t>(shape));
}

TEST_F(PaddleTensorTest, ctor_test) {
    Tensor t;
    // t holds no memory
    EXPECT_THROW({ PaddleTensor<int64_t> pt(&_cpu_ctx, t); }, ::paddle::platform::EnforceNotMet);
    t.template mutable_data<int64_t>(_cpu_ctx.GetPlace());
    EXPECT_NO_THROW({ PaddleTensor<int64_t> pt(&_cpu_ctx, t); });
}

TEST_F(PaddleTensorTest, shape_test) {
    std::vector<size_t> shape = { 2, 3 };
    auto pt = _tensor_factory->template create<int64_t>(shape);

    EXPECT_EQ(shape.size(), pt->shape().size());

    bool eq = std::equal(shape.begin(), shape.end(), pt->shape().begin());
    EXPECT_TRUE(eq);

    EXPECT_EQ(6u, pt->numel());
}

TEST_F(PaddleTensorTest, reshape_test) {
    std::vector<size_t> shape = { 2, 3 };
    auto pt = _tensor_factory->template create<int64_t>();

    pt->reshape(shape);

    EXPECT_EQ(shape.size(), pt->shape().size());

    bool eq = std::equal(shape.begin(), shape.end(), pt->shape().begin());
    EXPECT_TRUE(eq);
}

TEST_F(PaddleTensorTest, add_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    auto pt2 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 1;
    pt1->data()[0] = 2;
    pt0->add(pt1.get(), pt2.get());

    EXPECT_EQ(3, pt2->data()[0]);
}

TEST_F(PaddleTensorTest, sub_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    auto pt2 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 2;
    pt1->data()[0] = 1;
    pt0->sub(pt1.get(), pt2.get());

    EXPECT_EQ(1, pt2->data()[0]);
}

TEST_F(PaddleTensorTest, negative_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 2;
    pt0->negative(pt1.get());

    EXPECT_EQ(-2, pt1->data()[0]);
}

TEST_F(PaddleTensorTest, mul_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    auto pt2 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 7;
    pt1->data()[0] = 3;
    pt0->mul(pt1.get(), pt2.get());

    EXPECT_EQ(21, pt2->data()[0]);
}

TEST_F(PaddleTensorTest, div_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    auto pt2 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 7;
    pt1->data()[0] = 3;
    pt0->div(pt1.get(), pt2.get());

    EXPECT_EQ(2, pt2->data()[0]);
}

TEST_F(PaddleTensorTest, matmul_test) {
    std::vector<size_t> shape0 = { 2, 3 };
    std::vector<size_t> shape1 = { 3, 2 };
    std::vector<size_t> shape2 = { 2, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 6; ++i) {
        pt0->data()[i] = i;
        pt1->data()[i] = i;
    }
    pt0->mat_mul(pt1.get(), pt2.get());

    // | 0 1 2 |   | 0 1 |   | 10 13 |
    // | 3 4 5 | x | 2 3 | = | 28 40 |
    //             | 4 5 |

    std::vector<int64_t> res = { 10, 13, 28, 40 };

    bool eq = std::equal(res.begin(), res.end(), pt2->data());

    EXPECT_TRUE(eq);
}

TEST_F(PaddleTensorTest, matmul_transpose_test) {
    std::vector<size_t> shape0 = { 2, 3 };
    std::vector<size_t> shape1 = { 3, 2 };
    std::vector<size_t> shape2 = { 3, 3 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 6; ++i) {
        pt0->data()[i] = i;
        pt1->data()[i] = i;
    }
    pt0->mat_mul(pt1.get(), pt2.get(), true, true);

    // | 0 3 |   | 0 2 4 |   |  3  9 15 |
    // | 1 4 | x | 1 3 5 | = |  4 14 24 |
    // | 2 5 |               |  5 19 33 |

    std::vector<int64_t> res = { 3, 9, 15, 4, 14, 24, 5, 19, 33};

    bool eq = std::equal(res.begin(), res.end(), pt2->data());

    EXPECT_TRUE(eq);
}

TEST_F(PaddleTensorTest, matmul_transpose_test2) {
    std::vector<size_t> shape0 = { 2, 3 };
    std::vector<size_t> shape1 = { 2, 3 };
    std::vector<size_t> shape2 = { 2, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 6; ++i) {
        pt0->data()[i] = i;
        pt1->data()[i] = i;
    }
    pt0->mat_mul(pt1.get(), pt2.get(), false, true);

    // | 0 1 2 |   | 0 3 |   |  5 14 |
    // | 3 4 5 | x | 1 4 | = | 14 50 |
    //             | 2 5 |

    std::vector<int64_t> res = { 5, 14, 14, 50 };

    bool eq = std::equal(res.begin(), res.end(), pt2->data());

    EXPECT_TRUE(eq);
}

TEST_F(PaddleTensorTest, matmul_transpose_test3) {
    std::vector<size_t> shape0 = { 2, 3 };
    std::vector<size_t> shape1 = { 2, 3 };
    std::vector<size_t> shape2 = { 3, 3 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 6; ++i) {
        pt0->data()[i] = i;
        pt1->data()[i] = i;
    }
    pt0->mat_mul(pt1.get(), pt2.get(), true, false);

    // | 0 3 |   | 0 1 2 |   |  9 12 15 |
    // | 1 4 | x | 3 4 5 | = | 12 17 22 |
    // | 2 5 |               | 15 22 29 |

    std::vector<int64_t> res = { 9, 12, 15, 12, 17, 22, 15, 22, 29};

    bool eq = std::equal(res.begin(), res.end(), pt2->data());

    EXPECT_TRUE(eq);
}

TEST_F(PaddleTensorTest, matmul_test2) {
    std::vector<size_t> shape0 = { 2, 2, 3 };
    std::vector<size_t> shape1 = { 3, 2 };
    std::vector<size_t> shape2 = { 2, 2, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 12; ++i) {
        pt0->data()[i] = i;
        pt1->data()[i] = i;
    }
    pt0->mat_mul(pt1.get(), pt2.get());

    // | 0 1 2 | | 6  7  8 |    | 0 1 |   | 10 13 |  | 46 67 |
    // | 3 4 5 |,| 9 10 11 |  x | 2 3 | = | 28 40 |, | 64 94 |
    //                          | 4 5 |

    std::vector<int64_t> res = { 10, 13, 28, 40, 46, 67, 64, 94 };

    bool eq = std::equal(res.begin(), res.end(), pt2->data());

    EXPECT_TRUE(eq);
}

TEST_F(PaddleTensorTest, matmul_test3) {
    std::vector<size_t> shape0 = { 2, 2, 3 };
    std::vector<size_t> shape1 = { 2, 3, 2 };
    std::vector<size_t> shape2 = { 2, 2, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 12; ++i) {
        pt0->data()[i] = i;
        pt1->data()[i] = i;
    }
    pt0->mat_mul(pt1.get(), pt2.get());

    // | 0 1 2 | | 6  7  8 |    | 0 1 | |  6  7 |   | 10 13 |  | 172 193 |
    // | 3 4 5 |,| 9 10 11 |  x | 2 3 | |  8  9 | = | 28 40 |, | 244 274 |
    //                          | 4 5 |,| 10 11 |

    std::vector<int64_t> res = { 10, 13, 28, 40, 172, 193, 244, 274 };

    bool eq = std::equal(res.begin(), res.end(), pt2->data());

    EXPECT_TRUE(eq);
}

TEST_F(PaddleTensorTest, xor_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    auto pt2 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 3;
    pt1->data()[0] = 7;
    pt0->bitwise_xor(pt1.get(), pt2.get());

    EXPECT_EQ(4, pt2->data()[0]);
}

TEST_F(PaddleTensorTest, and_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    auto pt2 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 3;
    pt1->data()[0] = 7;
    pt0->bitwise_and(pt1.get(), pt2.get());

    EXPECT_EQ(3, pt2->data()[0]);
}

TEST_F(PaddleTensorTest, or_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    auto pt2 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 3;
    pt1->data()[0] = 7;
    pt0->bitwise_or(pt1.get(), pt2.get());

    EXPECT_EQ(7, pt2->data()[0]);
}

TEST_F(PaddleTensorTest, not_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 0;
    pt0->bitwise_not(pt1.get());

    EXPECT_EQ(-1, pt1->data()[0]);
}

TEST_F(PaddleTensorTest, lshift_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 2;
    pt0->lshift(1, pt1.get());

    EXPECT_EQ(4, pt1->data()[0]);
}

TEST_F(PaddleTensorTest, rshift_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = 2;
    pt0->rshift(1, pt1.get());

    EXPECT_EQ(1, pt1->data()[0]);
}

TEST_F(PaddleTensorTest, logical_rshift_test) {
    std::vector<size_t> shape = { 1 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape);
    auto pt1 = _tensor_factory->template create<int64_t>(shape);
    pt0->data()[0] = -1;
    pt0->logical_rshift(1, pt1.get());

    EXPECT_EQ(-1ull >> 1, pt1->data()[0]);
}


TEST_F(PaddleTensorTest, scale_test) {
    auto pt = _tensor_factory->template create<int64_t>();

    auto pt_ = dynamic_cast<PaddleTensor<int64_t>*>(pt.get());

    pt_->scaling_factor() = 1;

    Tensor t;

    int dim[1] = { 1 };
    paddle::framework::DDim ddim(dim, 1);
    t.template mutable_data<float>(ddim, _cpu_ctx.GetPlace());

    t.template data<float>()[0] = 0.25f;

    pt_->template from_float_point_type<float>(t, 2);

    EXPECT_EQ(2, pt_->scaling_factor());
    EXPECT_EQ(1, pt->data()[0]);
}

TEST_F(PaddleTensorTest, scalar_test) {
    auto pt = _tensor_factory->template create<int64_t>();

    auto pt_ = dynamic_cast<PaddleTensor<int64_t>*>(pt.get());

    pt_->scaling_factor() = 1;

    std::vector<size_t> shape = { 2 };
    pt_->template from_float_point_scalar(0.25f, shape, 2);

    EXPECT_EQ(2, pt_->scaling_factor());
    EXPECT_EQ(1, pt->data()[0]);
    EXPECT_EQ(1, pt->data()[1]);
}

TEST_F(PaddleTensorTest, slice_test) {
    std::vector<size_t> shape = { 2, 2 };
    auto pt = _tensor_factory->template create<int64_t>(shape);
    auto ret = _tensor_factory->template create<int64_t>();

    auto pt_ = dynamic_cast<PaddleTensor<int64_t>*>(pt.get());
    pt_->scaling_factor() = 1;

    for (size_t i = 0; i < 4; ++i) {
        pt->data()[0] = i;
    }

    pt_->slice(1, 2, ret.get());

    auto shape_ = ret->shape();

    EXPECT_EQ(2, shape_.size());
    EXPECT_EQ(1, shape_[0]);
    EXPECT_EQ(2, shape_[1]);

    EXPECT_EQ(1, ret->scaling_factor());

    EXPECT_EQ(2, ret->data()[0]);
    EXPECT_EQ(3, ret->data()[1]);
}

TEST_F(PaddleTensorTest, add128_test) {
    std::vector<size_t> shape0 = { 2, 2 };
    std::vector<size_t> shape1 = { 2, 2 };
    std::vector<size_t> shape2 = { 2, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 4; ++i) {
        pt0->data()[i] = 0;
        pt1->data()[i] = 0;
    }

    pt0->data()[2] = 1;
    pt1->data()[2] = 1;
    dynamic_cast<PaddleTensor<int64_t>*>(pt0.get())->add128(pt1.get(), pt2.get(), true, true);
    // | 0 1| + | 0 1| = | 0 2|
    std::vector<int64_t> res = { 0, 0, 0, 2};
    EXPECT_EQ(pt2->data()[0], 0);
    EXPECT_EQ(pt2->data()[1], 0);
    EXPECT_EQ(pt2->data()[2], 2);
    EXPECT_EQ(pt2->data()[3], 0);
}

TEST_F(PaddleTensorTest, sub128_test1) {
    std::vector<size_t> shape0 = { 2, 2 };
    std::vector<size_t> shape1 = { 2, 2 };
    std::vector<size_t> shape2 = { 2, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 4; ++i) {
        pt0->data()[i] = 0;
        pt1->data()[i] = 0;
    }

    pt0->data()[2] = 2;
    pt1->data()[2] = 1;
    dynamic_cast<PaddleTensor<int64_t>*>(pt0.get())->sub128(pt1.get(), pt2.get(), true, true);
    // | 0 2| - | 0 1| = | 0 1|
    EXPECT_EQ(pt2->data()[0], 0);
    EXPECT_EQ(pt2->data()[1], 0);
    EXPECT_EQ(pt2->data()[2], 1);
    EXPECT_EQ(pt2->data()[3], 0);
}

TEST_F(PaddleTensorTest, mul128_test1) {
    std::vector<size_t> shape0 = { 2, 2 };
    std::vector<size_t> shape1 = { 2, 2 };
    std::vector<size_t> shape2 = { 1, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 4; ++i) {
        pt0->data()[i] = 0;
        pt1->data()[i] = 0;
    }

    pt0->data()[2] = (int64_t) 2 << SCALING_FACTOR;
    pt1->data()[2] = (int64_t) 1 << SCALING_FACTOR;
    pt0->scaling_factor() = SCALING_FACTOR;

    dynamic_cast<PaddleTensor<int64_t>*>(pt0.get())->mul128_with_truncate(pt1.get(), pt2.get(), true, true);
    // | 0 2| * | 0 1| = | 0 2|
    EXPECT_EQ(pt2->data()[0], 0);
    EXPECT_EQ(pt2->data()[1], (int64_t) 2 << SCALING_FACTOR);
}

TEST_F(PaddleTensorTest, mul128_test2) {
    std::vector<size_t> shape0 = { 1, 2 };
    std::vector<size_t> shape1 = { 1, 2 };
    std::vector<size_t> shape2 = { 1, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 2; ++i) {
        pt0->data()[i] = 0;
        pt1->data()[i] = 0;
    }

    pt0->data()[1] = (int64_t) 2 << SCALING_FACTOR;
    pt1->data()[1] = (int64_t) 1 << SCALING_FACTOR;
    pt0->scaling_factor() = SCALING_FACTOR;

    dynamic_cast<PaddleTensor<int64_t>*>(pt0.get())->mul128_with_truncate(pt1.get(), pt2.get(), false, false);
    // | 0 2| * | 0 1| = | 0 2|
    EXPECT_EQ(pt2->data()[0], 0);
    EXPECT_EQ(pt2->data()[1], (int64_t) 2 << SCALING_FACTOR);
}

TEST_F(PaddleTensorTest, mul128_test3) {
    std::vector<size_t> shape0 = { 1, 2 };
    std::vector<size_t> shape1 = { 2, 2 };
    std::vector<size_t> shape2 = { 1, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 4; ++i) {
        pt1->data()[i] = 0;
    }

    pt0->data()[1] = (int64_t) 2 << SCALING_FACTOR;
    pt0->data()[0] = 0;
    pt1->data()[2] = (int64_t) 1 << SCALING_FACTOR;
    pt0->scaling_factor() = SCALING_FACTOR;

    dynamic_cast<PaddleTensor<int64_t>*>(pt0.get())->mul128_with_truncate(pt1.get(), pt2.get(), false, true);
    // | 0 2| * | 0 1| = | 0 2|
    EXPECT_EQ(pt2->data()[0], 0);
    EXPECT_EQ(pt2->data()[1], (int64_t) 2 << SCALING_FACTOR);
}

TEST_F(PaddleTensorTest, mul128_test4) {
    std::vector<size_t> shape0 = { 2, 2 };
    std::vector<size_t> shape1 = { 1, 2 };
    std::vector<size_t> shape2 = { 1, 2 };
    auto pt0 = _tensor_factory->template create<int64_t>(shape0);
    auto pt1 = _tensor_factory->template create<int64_t>(shape1);
    auto pt2 = _tensor_factory->template create<int64_t>(shape2);
    for (size_t i = 0; i < 4; ++i) {
        pt0->data()[i] = 0;
    }

    pt1->data()[0] = 0;
    pt1->data()[1] = (int64_t) 2 << SCALING_FACTOR;
    pt0->data()[2] = (int64_t) 1 << SCALING_FACTOR;
    pt0->scaling_factor() = SCALING_FACTOR;

    dynamic_cast<PaddleTensor<int64_t>*>(pt0.get())->mul128_with_truncate(pt1.get(), pt2.get(), true, false);
    // | 0 1| * | 0 2| = | 0 2|
    EXPECT_EQ(pt2->data()[0], 0);
    EXPECT_EQ(pt2->data()[1], (int64_t) 2 << SCALING_FACTOR);
}
} // namespace common
