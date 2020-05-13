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

#pragma once
#include "mpc_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

struct MpcGreaterThanFunctor {
    void Run(const Tensor *in_x_t, const Tensor *in_y_t, Tensor *out_t) {
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->gt(in_x_t, in_y_t, out_t);
    }
};

struct MpcGreaterEqualFunctor {
    void Run(const Tensor *in_x_t, const Tensor *in_y_t, Tensor *out_t) {
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->geq(in_x_t, in_y_t, out_t);
    }
};

struct MpcLessThanFunctor {
    void Run(const Tensor *in_x_t, const Tensor *in_y_t, Tensor *out_t) {
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->lt(in_x_t, in_y_t, out_t);
    }
};

struct MpcLessEqualFunctor {
    void Run(const Tensor *in_x_t, const Tensor *in_y_t, Tensor *out_t) {
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->leq(in_x_t, in_y_t, out_t);
    }
};

struct MpcEqualFunctor {
    void Run(const Tensor *in_x_t, const Tensor *in_y_t, Tensor *out_t) {
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->eq(in_x_t, in_y_t, out_t);
    }
};

struct MpcNotEqualFunctor {
    void Run(const Tensor *in_x_t, const Tensor *in_y_t, Tensor *out_t) {
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->neq(in_x_t, in_y_t, out_t);
    }
};

template <typename DeviceContext, typename T, typename Functor>
class MpcCompareOpKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override{
        auto *in_x_t = ctx.Input<framework::LoDTensor>("X");
        auto *in_y_t = ctx.Input<framework::LoDTensor>("Y");
        auto *out_t = ctx.Output<framework::LoDTensor>("Out");

        auto out = out_t->mutable_data<T>(ctx.GetPlace());
        Functor().Run(in_x_t, in_y_t, out_t);
    }
};
}  // namespace operators
}  // namespace paddl
