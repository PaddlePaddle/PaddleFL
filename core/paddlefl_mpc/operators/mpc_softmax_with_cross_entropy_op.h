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
using DDim = framework::DDim;

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}


// Out = softmax(Logits) = relu(Logits_i) / sum(relu(Logits_i)): prediction of input.
// todo: loss=?
template <typename DeviceContext, typename T>
class MpcSoftmaxWithCrossEntropyKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *in_x_t = ctx.Input<Tensor>("Logits");
        auto *out_softmax_t = ctx.Output<Tensor>("Softmax");
        auto *out_loss_t = ctx.Output<Tensor>("Loss");
        out_softmax_t->mutable_data<T>(ctx.GetPlace());
        out_loss_t->mutable_data<T>(ctx.GetPlace());
        bool use_relu = ctx.Attr<bool>("use_relu");
        bool use_long_div = ctx.Attr<bool>("use_long_div");

        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->softmax(
            in_x_t, out_softmax_t, use_relu, use_long_div);
    }
};

template <typename DeviceContext, typename T>
struct SetExpandData {
    void operator()(T* dst, const T* src, size_t n, size_t d);
};

// dx = dout.expand * (softmax(x) - labels)
template <typename DeviceContext, typename T>
class MpcSoftmaxWithCrossEntropyGradKernel : public MpcOpKernel<T> {
public:
    void ComputeImpl(const framework::ExecutionContext &ctx) const override {
        auto *dout = ctx.Input<Tensor>(framework::GradVarName("Loss"));
        auto *in_label_t = ctx.Input<Tensor>("Label");
        auto *in_softmax_t = ctx.Input<Tensor>("Softmax");
        auto *dx = ctx.Output<Tensor>(framework::GradVarName("Logits"));
        const bool soft_label = ctx.Attr<bool>("soft_label");
        PADDLE_ENFORCE_EQ(soft_label, true, "soft_label can only be true.");

        const int rank = dx->dims().size();
        const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
        int axis_dim = dx->dims()[axis];
        const int n = SizeToAxis(axis, dx->dims());
        const int d = SizeFromAxis(axis, dx->dims());

        T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
        const T* dout_data = dout->data<T>();

        // expand dout
        Tensor dout_expand;
        T* dout_expand_data = dout_expand.mutable_data<T>(dx->dims(), ctx.GetPlace());

        auto set_expand_functor = SetExpandData<DeviceContext, T>();
        set_expand_functor(dout_expand_data, dout_data, n, d);

        // dx = dout.expand * (softmax - label)
        Tensor softmax_minus_label;
        softmax_minus_label.mutable_data<T>(in_label_t->dims(), ctx.GetPlace());
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->sub(in_softmax_t, in_label_t, &softmax_minus_label);
        mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators()->elementwise_mul(&dout_expand, &softmax_minus_label, dx);
    }
};

}  // namespace operators
}  // namespace paddle
