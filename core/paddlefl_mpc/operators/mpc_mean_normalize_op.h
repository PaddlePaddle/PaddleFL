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

#include <algorithm>

#include "paddle/fluid/framework/op_registry.h"
#include "mpc_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MpcMeanNormalizationKernel : public MpcOpKernel<T> {
 public:
  void ComputeImpl(const framework::ExecutionContext& context) const override {
    const Tensor* min = context.Input<Tensor>("Min");
    const Tensor* max = context.Input<Tensor>("Max");
    const Tensor* mean = context.Input<Tensor>("Mean");
    const Tensor* sample_num = context.Input<Tensor>("SampleNum");
    const Tensor* total_num = context.Input<Tensor>("TotalNum");

    Tensor* range = context.Output<Tensor>("Range");
    Tensor* mean_out = context.Output<Tensor>("MeanOut");

    int share_num = min->dims()[0];
    int party_num = min->dims()[1];
    int feat_num = min->dims()[2];

    Tensor neg_min;
    neg_min.mutable_data<T>(min->dims(), context.GetPlace(), 0);

    Tensor neg_min_global;
    Tensor max_global;

    neg_min_global.mutable_data<T>(
        framework::make_ddim({share_num, 1, feat_num}), context.GetPlace(), 0);
    max_global.mutable_data<T>(
        framework::make_ddim({share_num, 1, feat_num}), context.GetPlace(), 0);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->neg(min, &neg_min);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->max(&neg_min, &neg_min_global);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->max(max, &max_global);

    range->mutable_data<T>(
        framework::make_ddim({share_num, 1, feat_num}), context.GetPlace(), 0);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->add(&max_global, &neg_min_global, range);

    range->mutable_data<T>(
        framework::make_ddim({share_num, feat_num}), context.GetPlace(), 0);

    Tensor sample_num_;

    sample_num_.ShareDataWith(*sample_num);

    sample_num_.mutable_data<T>(
        framework::make_ddim({share_num, 1, party_num}), context.GetPlace(), 0);

    mean_out->mutable_data<T>(
        framework::make_ddim({share_num, 1, feat_num}), context.GetPlace(), 0);

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->matmul(&sample_num_, mean, mean_out);

    mean_out->mutable_data<T>(
        framework::make_ddim({share_num, feat_num}), context.GetPlace(), 0);

    Tensor total_num_;

    total_num_.mutable_data<T>(
        framework::make_ddim({share_num, feat_num}), context.GetPlace(), 0);

    // broadcasting total_num to shape [share_num, feat_num]
    for (int i = 0; i < share_num; ++i) {
        std::fill(total_num_.data<T>() + i * feat_num,
                  total_num_.data<T>() + (i + 1) * feat_num,
                  total_num->data<T>()[i]);
    }

    mpc::MpcInstance::mpc_instance()->mpc_protocol()
        ->mpc_operators()->div(mean_out, &total_num_, mean_out);

}
};

}  // namespace operators
}  // namespace paddle
