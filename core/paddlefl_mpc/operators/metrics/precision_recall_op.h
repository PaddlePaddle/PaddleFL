/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "../mpc_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MpcPrecisionRecallKernel : public MpcOpKernel<T> {
 public:
  void ComputeImpl(const framework::ExecutionContext& context) const override {
    const Tensor* preds = context.Input<Tensor>("Predicts");
    const Tensor* lbls = context.Input<Tensor>("Labels");
    const Tensor* stats = context.Input<Tensor>("StatesInfo");
    Tensor* batch_metrics = context.Output<Tensor>("BatchMetrics");
    Tensor* accum_metrics = context.Output<Tensor>("AccumMetrics");
    Tensor* accum_stats = context.Output<Tensor>("AccumStatesInfo");


    float threshold = context.Attr<float>("threshold");

    Tensor idx;
    idx.mutable_data<T>(preds->dims(), context.GetPlace(), 0);

    Tensor batch_stats;
    batch_stats.mutable_data<T>(stats->dims(), context.GetPlace(), 0);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->predicts_to_indices(preds, &idx, threshold);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->calc_tp_fp_fn(&idx, lbls, &batch_stats);

    batch_metrics->mutable_data<T>(framework::make_ddim({3}), context.GetPlace(), 0);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->calc_precision_recall(&batch_stats, batch_metrics);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->add(&batch_stats, stats, accum_stats);

    accum_metrics->mutable_data<T>(framework::make_ddim({3}), context.GetPlace(), 0);
    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->calc_precision_recall(accum_stats, accum_metrics);
  }
};

}  // namespace operators
}  // namespace paddle
