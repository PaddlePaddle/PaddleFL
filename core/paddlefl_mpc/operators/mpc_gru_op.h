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
#include <string>
#include <functional>
#include <glog/logging.h>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "core/paddlefl_mpc/operators/math/sequence2batch.h"
#include "core/paddlefl_mpc/operators/math/concat_and_split.h"
#include "core/paddlefl_mpc/operators/math/math_function.h"
#include "mpc_op.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;
typedef std::function<void(const Tensor*, Tensor*)> GateActivation;

template<typename T>
inline void ComputeSigmoidGrad(const framework::ExecutionContext& context,
                               Tensor& dy, Tensor& y, Tensor& dx);
template<typename DeviceContext, typename T>
inline void BackwardStateGrad(const framework::ExecutionContext& context,
                              std::vector<Tensor>& mpc_splitted_gate_t,
                              std::vector<Tensor>& mpc_splitted_gate_grad_t,
                              Tensor& mpc_hidden_prev_t, Tensor& mpc_hidden_prev_grad_t,
                              Tensor& mpc_hidden_grad_t,
                              bool origin_mode, bool has_hidden_prev,
                              bool has_hidden_prev_grad);

template<typename DeviceContext, typename T>
inline void BackwarsResetGrad(const framework::ExecutionContext& context,
                              std::vector<Tensor>& mpc_splitted_gate_t,
                              std::vector<Tensor>& mpc_splitted_gate_grad_t,
                              Tensor& mpc_hidden_prev_t, Tensor& mpc_hidden_prev_grad_t,
                              Tensor& mpc_reset_hidden_prev_grad_t,
                              bool has_hidden_prev, bool has_hidden_prev_grad);

template <typename DeviceContext, typename T>
inline void ReorderInitState(const DeviceContext& ctx,
                             const framework::Tensor& src,
                             framework::Vector<size_t> index_lod,
                             framework::Tensor* dst, bool indexed_src) {
    math::CopyMatrixRowsFunctor<DeviceContext, T> row_shuffle;
    dst->mutable_data<T>(src.dims(), ctx.GetPlace());
    row_shuffle(ctx, src, index_lod, dst, indexed_src);
}

template<typename DeviceContext, typename T>
inline void ComputGRUUint(const framework::ExecutionContext& context,
                          std::vector<Tensor>& gate_t,
                          std::vector<Tensor>& weight_t,
                          Tensor &reset_hidden_prev_t,
                          Tensor &hidden_t,
                          Tensor &hidden_prev_t,
                          bool origin_mode,
                          bool& has_hidden_prev) {
    // compute GRUUnit
    Tensor u_h_t;
    Tensor r_h_t;
    // gate_t[x] shape (2, B, D)
    // weight_t[x] shape (2, D, D)
    // hidden_prev_t shape (2, B, D)
    // hidden_t shape (2, B, D)
    u_h_t.mutable_data<T>(gate_t[0].dims(), context.GetPlace());
    r_h_t.mutable_data<T>(gate_t[1].dims(), context.GetPlace());
    auto mpc_operator = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
    if (has_hidden_prev) {
        // compute update gate and reset gate: gate_t += hidden_prev_t matmul gate_weight
        mpc_operator->matmul(&hidden_prev_t, &weight_t[0], &u_h_t);
        mpc_operator->add(&u_h_t, &gate_t[0], &gate_t[0]);

        mpc_operator->matmul(&hidden_prev_t, &weight_t[1], &r_h_t);
        mpc_operator->add(&r_h_t, &gate_t[1], &gate_t[1]);
    }

    auto GateActProcess = [&gate_t](const GateActivation fun) {
        fun(&gate_t[0], &gate_t[0]);
        fun(&gate_t[1], &gate_t[1]);
    };
    GateActivation activ_functor;
    std::string active_gate = context.Attr<std::string>("gate_activation");
    if (active_gate == "sigmoid_chebyshev") {
        activ_functor = std::bind(&paddle::mpc::MpcOperators::sigmoid_chebyshev,
                                  mpc_operator.get(),
                                  std::placeholders::_1,
                                  std::placeholders::_2);
    } else if (active_gate == "sigmoid") {
        activ_functor = std::bind(&paddle::mpc::MpcOperators::sigmoid,
                                  mpc_operator.get(),
                                  std::placeholders::_1,
                                  std::placeholders::_2);
    } else if (active_gate == "sigmoid_enhanced") {
        activ_functor = std::bind(&paddle::mpc::MpcOperators::sigmoid_enhanced,
                                  mpc_operator.get(),
                                  std::placeholders::_1,
                                  std::placeholders::_2);
    } else {
        PADDLE_THROW("gate activation of %s is not implemented yet.", active_gate);
    }
    GateActProcess(activ_functor);

    if (has_hidden_prev) {
        // reset_hidden_prev_t = gate[1] * hidden_prev_t
        // compute candidate gate: gate_t[2] += reset_hidden_prev_t matmul state_weight
        Tensor r_h_tmp;
        r_h_tmp.mutable_data<T>(gate_t[2].dims(), context.GetPlace());
        mpc_operator->mul(&gate_t[1], &hidden_prev_t, &reset_hidden_prev_t);
        mpc_operator->matmul(&reset_hidden_prev_t, &weight_t[2], &r_h_tmp);
        mpc_operator->add(&r_h_tmp, &gate_t[2], &gate_t[2]);
    } else {
        //initialize reset_hidden_prev_t and hidden_prev_t as 0
        math::SetConstant<DeviceContext, T> zero;
        auto& dev_ctx = context.template device_context<DeviceContext>();
        reset_hidden_prev_t.mutable_data<T>(gate_t[0].dims(), context.GetPlace());
        hidden_prev_t.mutable_data<T>(gate_t[0].dims(), context.GetPlace());
        zero(dev_ctx, &reset_hidden_prev_t, static_cast<T>(0));
        zero(dev_ctx, &hidden_prev_t, static_cast<T>(0));
        has_hidden_prev = true;
    }

    mpc_operator->relu(&gate_t[2], &gate_t[2]);

    Tensor u_h_tmp;
    Tensor ops_u_h_tmp;
    u_h_tmp.mutable_data<T>(hidden_t.dims(), context.GetPlace());
    ops_u_h_tmp.mutable_data<T>(hidden_t.dims(), context.GetPlace());
    if (origin_mode) {
        // compute output hidden_t = (gate[0] * hidden_prev_t + gate[2] - gate[0] * gate[2])
        mpc_operator->mul(&gate_t[0], &hidden_prev_t, &u_h_tmp);
        mpc_operator->add(&gate_t[2], &u_h_tmp, &u_h_tmp);
        mpc_operator->mul(&gate_t[0], &gate_t[2], &ops_u_h_tmp);
        mpc_operator->sub(&u_h_tmp, &ops_u_h_tmp, &hidden_t);
    } else {
        // compute output hidden_t = (gate[0] * gate[2] + hidden_prev_t - gate[0] * hidden_prev_t)
        mpc_operator->mul(&gate_t[0], &gate_t[2], &u_h_tmp);
        mpc_operator->add(&hidden_prev_t, &u_h_tmp, &u_h_tmp);
        mpc_operator->mul(&gate_t[0], &hidden_prev_t, &ops_u_h_tmp);
        mpc_operator->sub(&u_h_tmp, &ops_u_h_tmp, &hidden_t);
    }
}

inline void SliceAndReshape(const Tensor* input, Tensor &output, int i) {
    // Slice mpc tensor to share[i]
    output = input->Slice(i, i + 1);
    auto dims = output.dims();
    output.Resize(paddle::framework::slice_ddim(dims, 1, dims.size()));
}

template<typename DeviceContext, typename T>
inline void ToMpcBatchTensor(const framework::ExecutionContext& context,
                             Tensor& output, const Tensor& input,
                             int start, int end) {
    //input : (2 , T, x) -> output: (2, end - start, x)
    auto dims = input.dims();
    auto& dev_ctx = context. template device_context<DeviceContext>();
    math::Transpose<DeviceContext, T, 3> transpose;
    Tensor tmp;
    tmp.mutable_data<T>(framework::make_ddim({dims[1], dims[0], dims[2]}), context.GetPlace());
    transpose(dev_ctx, input, &tmp, {1, 0, 2});
    Tensor tmp_slice = tmp.Slice(start, end);
    output.mutable_data<T>(framework::make_ddim({dims[0], end - start, dims[2]}), context.GetPlace());
    transpose(dev_ctx, tmp_slice, &output, {1, 0, 2});
}

template<typename DeviceContext, typename T>
inline void Split3Dim(const framework::ExecutionContext& context,
                      std::vector<Tensor>& output,
                      const Tensor& input) {
    // input : (2, x, 3D) -> output : 3 * (2, x, D)
    auto& dev_ctx = context. template device_context<DeviceContext>();
    Tensor tmp_trans;
    auto dims = input.dims();
    int frame_size = dims[2] / 3;
    tmp_trans.mutable_data<T>(framework::make_ddim({dims[2], dims[0], dims[1]}), context.GetPlace());
    math::Transpose<DeviceContext, T, 3> transpose;
    transpose(dev_ctx, input, &tmp_trans, {2, 0, 1});
    for (int i = 0; i < 3; ++i) {
        Tensor tmp_slice = tmp_trans.Slice(i * frame_size, (i + 1) * frame_size);
        Tensor tmp_re_trans;
        tmp_re_trans.mutable_data<T>(framework::make_ddim({dims[0], dims[1], dims[2] / 3}),
                                      context.GetPlace());
        transpose(dev_ctx, tmp_slice, &tmp_re_trans, {1, 2, 0});
        output.emplace_back(tmp_re_trans);
    }
}


template<typename DeviceContext, typename T>
inline void Concat3Dim(const framework::ExecutionContext& context,
                       Tensor* output,
                       std::vector<Tensor>& input) {
    // input 3 * (2, x, D) -> (2, x, 3D)
    math::ConcatFunctor<DeviceContext, T> concat;
    auto& input_dims = input[0].dims();
    std::vector<int64_t> output_dim{input_dims[0], input_dims[1], input_dims[2] * 3};
    output->mutable_data<T>(framework::make_ddim(output_dim), context.GetPlace());
    auto& dev_ctx = context. template device_context<DeviceContext>();
    concat(dev_ctx, input, 3, output);
}

template<typename DeviceContext, typename T>
inline void SplitWeight(const framework::ExecutionContext& context,
                      std::vector<Tensor>& splitted_weights,
                      const Tensor& weight) {
    // split weight[0]、weight[1]、weight[2] with shape (2, D, D) from weight(2, D, 3D)
    // note that weight[2]'s data start at offset 2 * D * D of weight's data
    auto& dev_ctx = context. template device_context<DeviceContext>();
    auto dims = weight.dims();
    auto frame_size = dims[2] / 3;
    splitted_weights.resize(3);
    auto place = context.GetPlace();

    // copy weight[0] weight[1] from weight
    Tensor update_weight;
    update_weight.mutable_data<T>(framework::make_ddim({2, frame_size, 2 * frame_size}),
                                      place);
    //splitted_weights->at(2) = new Tensor();
    splitted_weights[2].mutable_data<T>(framework::make_ddim({2, frame_size, frame_size}),
                                             place);
    for (int i = 0; i < 2; ++i) {
        Tensor weight_s;
        Tensor update_weight_s;
        Tensor weight_3_s;
        SliceAndReshape(&weight, weight_s, i);
        SliceAndReshape(&update_weight, update_weight_s, i);
        SliceAndReshape(&splitted_weights[2], weight_3_s, i);
        T* update_s_data = update_weight_s.mutable_data<T>(place);
        T* weight_s_data = weight_s.data<T>();
        memcpy(update_s_data, weight_s_data, update_weight_s.numel() * sizeof(T));
        // weight[3]
        memcpy(weight_3_s.mutable_data<T>(place), weight_s_data + 2 * frame_size * frame_size,
               weight_3_s.numel() * sizeof(T));
    }
    // split update_weight to weight[0] and weight[1]
    math::Transpose<DeviceContext, T, 3> transpose;
    Tensor weight_trans;
    weight_trans.mutable_data<T>(framework::make_ddim({2 * frame_size, 2, frame_size}), place);
    transpose(dev_ctx, update_weight, &weight_trans, {2, 0, 1});
    for (int i = 0; i < 2; ++i) {
        //splitted_weights->at(i) = new Tensor();
        splitted_weights[i].mutable_data<T>(framework::make_ddim({2, frame_size, frame_size}), place);
        transpose(dev_ctx, weight_trans.Slice(frame_size * i, frame_size * (i + 1)),
                  &splitted_weights[i], {1, 2, 0});
    }
}

template<typename DeviceContext, typename T>
inline void ConcatWeight(const framework::ExecutionContext& context,
                       Tensor* weight,
                       std::vector<Tensor>& splitted_weights) {
    // concat weight[0]、weight[1]、weight[2] with shape (2, D, D) to weight(2, D, 3D)
    // note that weight[2]'s data append after weight[0] and weight[1]
    // weight[0] and weight[1] are concat as shape (2, D, 2D) in axis 2
    math::ConcatFunctor<DeviceContext, T> concat;
    std::vector<Tensor> update_weight_list;
    update_weight_list.resize(2);
    auto place = context.GetPlace();
    auto& splitted_weights_dims = splitted_weights[0].dims();
    std::vector<int64_t> weight_dim{splitted_weights_dims[0], splitted_weights_dims[1],
                                    splitted_weights_dims[2] * 3};
    weight->mutable_data<T>(framework::make_ddim(weight_dim), context.GetPlace());
    for (int i = 0; i < 2; ++i) {
        update_weight_list[i] = splitted_weights[i];
    }
    auto& dev_ctx = context. template device_context<DeviceContext>();
    // Concat update weight and reset weight as update weights
    Tensor update_weights;
    update_weights.mutable_data<T>(
        framework::make_ddim({splitted_weights_dims[0],
                splitted_weights_dims[1],
                splitted_weights_dims[2] * 2}),
        place);
    concat(dev_ctx, update_weight_list, 3, &update_weights);
    // Concat candidate weight
    for (int i = 0; i < 2; ++i) {
        Tensor weight_s = weight->Slice(i, i + 1);
        Tensor update_weights_s = update_weights.Slice(i, i + 1);
        Tensor reset_weight_s = splitted_weights[i].Slice(i, i + 1);

        T* weight_s_data = weight_s.mutable_data<T>(place);
        T* update_weights_s_data = update_weights_s.data<T>();
        T* reset_weight_s_data = reset_weight_s.data<T>();

        size_t numel_update = update_weights_s.numel();
        memcpy(weight_s_data, update_weights_s_data, numel_update * sizeof(T));
        memcpy(weight_s_data + numel_update, reset_weight_s_data, reset_weight_s.numel());
    }
}

template<typename DeviceContext, typename T>
inline void ConcatBatchOne(const framework::ExecutionContext& context,
                           Tensor* output,
                           Tensor& input,
                           int start,
                           int end) {
    // replace output[2, start:end, x] with input (2, end - start, x)

    auto& dev_ctx = context. template device_context<DeviceContext>();
    Tensor tmp_trans;
    auto dims = output->dims();
    tmp_trans.mutable_data<T>(framework::make_ddim({dims[1], dims[0], dims[2]}), context.GetPlace());
    math::Transpose<DeviceContext, T, 3> transpose;
    transpose(dev_ctx, *output, &tmp_trans, {1, 0, 2});
    Tensor splitted_t0;
    Tensor splitted_t2;
    Tensor splitted_t0_rec;
    Tensor splitted_t2_rec;
    std::vector<Tensor> concat_in;
    if (start > 0) {
        splitted_t0 = tmp_trans.Slice(0, start);
        auto t0_dims = splitted_t0.dims();
        splitted_t0_rec.mutable_data<T>(framework::make_ddim({t0_dims[1], t0_dims[0], t0_dims[2]}),
                                    context.GetPlace());
        transpose(dev_ctx, splitted_t0, &splitted_t0_rec, {1, 0, 2});
        concat_in.emplace_back(splitted_t0_rec);
    }
    concat_in.emplace_back(input);
    if (end < dims[1]) {
        splitted_t2 = tmp_trans.Slice(end, dims[1]);
        auto t2_dims = splitted_t2.dims();
        splitted_t2_rec.mutable_data<T>(framework::make_ddim({t2_dims[1], t2_dims[0], t2_dims[2]}),
                                    context.GetPlace());
        transpose(dev_ctx, splitted_t2, &splitted_t2_rec, {1, 0, 2});
        concat_in.emplace_back(splitted_t2_rec);
    }

    math::ConcatFunctor<DeviceContext, T> concat;
    concat(dev_ctx, concat_in, 1, output);
}

template<typename DeviceContext, typename T>
inline void ConcatBatchAll(const framework::ExecutionContext& context,
                           Tensor* output,
                           std::vector<Tensor>& input) {
    // Concat all input tensors in dims[1]
    math::ConcatFunctor<DeviceContext, T> concat;
    auto& dev_ctx = context. template device_context<DeviceContext>();
    concat(dev_ctx, input, 1, output);
}

template<typename DeviceContext, typename T>
inline void GRUUnitGradCompute(const framework::ExecutionContext& context,
                               std::vector<Tensor>& mpc_splitted_gate_t,
                               std::vector<Tensor>& mpc_splitted_gate_grad_t,
                               Tensor& mpc_hidden_prev_t, Tensor& mpc_hidden_prev_grad_t,
                               std::vector<Tensor>& mpc_splitted_weights_t,
                               std::vector<Tensor>& mpc_splitted_weights_grad_t,
                               Tensor& mpc_reset_hidden_prev_t, Tensor& mpc_reset_hidden_prev_grad_t,
                               Tensor& mpc_hidden_grad_t, bool origin_mode,
                               bool& has_hidden_prev, bool& has_hidden_prev_grad,
                               bool& has_weight_grad) {
    // compute GRUUnitGrad
    BackwardStateGrad<DeviceContext, T>(context,
                         mpc_splitted_gate_t, mpc_splitted_gate_grad_t,
                         mpc_hidden_prev_t, mpc_hidden_prev_grad_t,
                         mpc_hidden_grad_t,
                         origin_mode, has_hidden_prev, has_hidden_prev_grad);
    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol,
                            "Protocol %s is not yet created in MPC Protocol.");
    auto mpc_operator = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
    math::Transpose<DeviceContext, T, 3> transpose;
    auto& dev_ctx = context. template device_context<DeviceContext>();
    std::vector<int> trans_axis{0, 2, 1};
    if (has_hidden_prev && has_hidden_prev_grad) {
        auto res_hidden_dims = mpc_reset_hidden_prev_grad_t.dims();
        // (B, D) * (D, D)^T + (B, D) :
        //reset_hidden_prev_grad = batch_gate_grad[2] * state_weight[2] + reset_hidden_prev_grad
        Tensor tmp;
        tmp.mutable_data<T>(res_hidden_dims, context.GetPlace());

        transpose(dev_ctx, mpc_splitted_weights_t[2], &tmp, trans_axis);
        mpc_operator->matmul(&mpc_splitted_gate_t[2], &tmp, &tmp);
        mpc_operator->add(&mpc_reset_hidden_prev_grad_t, &tmp, &mpc_reset_hidden_prev_grad_t);

        if (has_weight_grad) {
            // (B, D)^T * (B, D) + (D, D)
            // state_weight_grad[2] = reset_hidden_prev * batch_gate_grad[2] + state_weight_grad[2]
            Tensor tmp1, tmp2;
            tmp1.mutable_data<T>(
                framework::make_ddim(
                    std::vector<int64_t>({res_hidden_dims[0], res_hidden_dims[2], res_hidden_dims[1]})),
                context.GetPlace());
            tmp2.mutable_data<T>(mpc_splitted_weights_t[2].dims(), context.GetPlace());
            transpose(dev_ctx, mpc_reset_hidden_prev_t, &tmp1, trans_axis);
            mpc_operator->matmul(&tmp1, &mpc_splitted_gate_grad_t[2], &tmp2);
            mpc_operator->add(&mpc_splitted_weights_grad_t[2], &tmp2, &mpc_splitted_weights_grad_t[2]);
        }
    }
    BackwarsResetGrad<DeviceContext, T>(context,
                         mpc_splitted_gate_t, mpc_splitted_gate_grad_t,
                         mpc_hidden_prev_t, mpc_hidden_prev_grad_t,
                         mpc_reset_hidden_prev_grad_t,
                         has_hidden_prev, has_hidden_prev_grad);
    if (has_hidden_prev && has_hidden_prev_grad) {
        // (B, 2D) * (D, 2D)^T + (B, D)
        // hidden_prev_grad = batch_gate_grad * gate_weight + hidden_prev_grad
        // block matrix multiplication: A=[block_A1, block_A2], B^T=[block_B1, block_B2]
        // A*B = block_A1*block_B1 + block_A2*block_B2
        Tensor tmp1, tmp2;
        tmp1.mutable_data<T>(mpc_splitted_weights_t[0].dims(), context.GetPlace());
        tmp2.mutable_data<T>(mpc_hidden_prev_t.dims(), context.GetPlace());
        transpose(dev_ctx, mpc_splitted_weights_t[0], &tmp1, trans_axis);
        mpc_operator->matmul(&mpc_splitted_gate_grad_t[0], &tmp1, &tmp2);
        mpc_operator->add(&mpc_hidden_prev_grad_t, &tmp2, &mpc_hidden_prev_grad_t);

        transpose(dev_ctx, mpc_splitted_weights_t[1], &tmp1, trans_axis);
        mpc_operator->matmul(&mpc_splitted_gate_grad_t[1], &tmp1, &tmp2);
        mpc_operator->add(&mpc_hidden_prev_grad_t, &tmp2, &mpc_hidden_prev_grad_t);

        if (has_weight_grad) {
            // (B, D)^T * (B, 2D) + (D, 2D)
            // gate_weight_grad = hidden_prev * batch_gate_grad + gate_weight_grad
            auto hid_dims = mpc_hidden_prev_t.dims();
            Tensor tmp3, tmp4;
            tmp3.mutable_data<T>(
                framework::make_ddim({hid_dims[0], hid_dims[2], hid_dims[1]}),
                context.GetPlace());
            tmp4.mutable_data<T>(mpc_splitted_weights_t[0].dims(), context.GetPlace());
            transpose(dev_ctx, mpc_hidden_prev_t, &tmp3, trans_axis);
            mpc_operator->matmul(&tmp3, &mpc_splitted_gate_grad_t[0], &tmp4);
            mpc_operator->add(&mpc_splitted_weights_grad_t[0], &tmp4, &mpc_splitted_weights_grad_t[0]);

            mpc_operator->matmul(&tmp3, &mpc_splitted_gate_grad_t[1], &tmp4);
            mpc_operator->add(&mpc_splitted_weights_grad_t[1], &tmp4, &mpc_splitted_weights_grad_t[1]);
        }
    }
}

template<typename DeviceContext, typename T>
inline void BackwardStateGrad(const framework::ExecutionContext& context,
                              std::vector<Tensor>& mpc_splitted_gate_t,
                              std::vector<Tensor>& mpc_splitted_gate_grad_t,
                              Tensor& mpc_hidden_prev_t, Tensor& mpc_hidden_prev_grad_t,
                              Tensor& mpc_hidden_grad_t,
                              bool origin_mode, bool has_hidden_prev,
                              bool has_hidden_prev_grad) {
    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol,
                            "Protocol %s is not yet created in MPC Protocol.");
    auto mpc_operator = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
    math::SetConstant<DeviceContext, T> zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (!has_hidden_prev) {
        zero(dev_ctx, &mpc_hidden_prev_t, static_cast<T>(0));
    }
    if (!has_hidden_prev_grad) {
        zero(dev_ctx, &mpc_hidden_prev_grad_t, static_cast<T>(0));
    }

    if (origin_mode) {
        // batch_gate_grad[0] = hidden_grad * (hidden_prev - batch_gate[2])
        mpc_operator->sub(&mpc_hidden_prev_t, &mpc_splitted_gate_t[2], &mpc_splitted_gate_grad_t[0]);
        mpc_operator->mul(&mpc_hidden_grad_t, &mpc_splitted_gate_grad_t[0], &mpc_splitted_gate_grad_t[0]);
        // hidden_prev_grad += hidden_grad * batch_gate[0]
        Tensor tmp;
        tmp.mutable_data<T>(mpc_hidden_prev_grad_t.dims(), context.GetPlace());
        mpc_operator->mul(&mpc_hidden_grad_t, &mpc_splitted_gate_t[0], &tmp);
        mpc_operator->add(&mpc_hidden_prev_grad_t, &tmp, &mpc_hidden_prev_grad_t);

        // batch_gate_grad[2] = activation(hidden_grad * (1-batch_gate[0]), batch_gate[2])
        // activation = grad_relu (return a * (b > 0.0 ? 1.0 : 0.0);)
        Tensor tmp1;
        tmp1.mutable_data<T>(mpc_splitted_gate_grad_t[2].dims(), context.GetPlace());
        mpc_operator->mul(&mpc_hidden_grad_t, &mpc_splitted_gate_t[0], &tmp1);
        mpc_operator->sub(&mpc_hidden_grad_t, &tmp1, &tmp1);
        mpc_operator->relu_grad(&mpc_splitted_gate_t[2], &tmp1, &mpc_splitted_gate_grad_t[2], 0);

    } else {
        // batch_gate_grad[0] = hidden_grad * (batch_gate[2] - hidden_prev)
        mpc_operator->sub(&mpc_splitted_gate_t[2], &mpc_hidden_prev_t, &mpc_splitted_gate_grad_t[0]);
        mpc_operator->mul(&mpc_hidden_grad_t, &mpc_splitted_gate_grad_t[0], &mpc_splitted_gate_grad_t[0]);
        // hidden_prev_grad += hidden_grad * (1 - batch_gate[0])
        Tensor tmp;
        tmp.mutable_data<T>(mpc_hidden_prev_grad_t.dims(), context.GetPlace());
        mpc_operator->mul(&mpc_hidden_grad_t, &mpc_splitted_gate_t[0], &tmp);
        mpc_operator->sub(&mpc_hidden_grad_t, &tmp, &tmp);
        mpc_operator->add(&mpc_hidden_prev_grad_t, &tmp, &mpc_hidden_prev_grad_t);

        // batch_gate_grad[2] = activation(hidden_grad*batch_gate[0], batch_gate[2])
        // activation = grad_relu
        Tensor tmp1;
        tmp1.mutable_data<T>(mpc_splitted_gate_grad_t[2].dims(), context.GetPlace());
        mpc_operator->mul(&mpc_hidden_grad_t, &mpc_splitted_gate_t[0], &tmp1);
        mpc_operator->relu_grad(&mpc_splitted_gate_t[2], &tmp1, &mpc_splitted_gate_grad_t[2], 0);
    }
}

template<typename DeviceContext, typename T>
inline void BackwarsResetGrad(const framework::ExecutionContext& context,
                              std::vector<Tensor>& mpc_splitted_gate_t,
                              std::vector<Tensor>& mpc_splitted_gate_grad_t,
                              Tensor& mpc_hidden_prev_t, Tensor& mpc_hidden_prev_grad_t,
                              Tensor& mpc_reset_hidden_prev_grad_t,
                              bool has_hidden_prev, bool has_hidden_prev_grad) {
    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol,
                            "Protocol %s is not yet created in MPC Protocol.");
    auto mpc_operator = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
    math::SetConstant<DeviceContext, T> zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (!has_hidden_prev) {
        zero(dev_ctx, &mpc_hidden_prev_t, static_cast<T>(0));
    }
    if (!has_hidden_prev_grad) {
        zero(dev_ctx, &mpc_hidden_prev_grad_t, static_cast<T>(0));
    }
    if (!has_hidden_prev || !has_hidden_prev_grad) {
        zero(dev_ctx, &mpc_reset_hidden_prev_grad_t, static_cast<T>(0));
    }
    // batch_gate_grad[1] = reset_hidden_grad * hidden_prev
    mpc_operator->mul(&mpc_reset_hidden_prev_grad_t, &mpc_hidden_prev_t, &mpc_splitted_gate_grad_t[1]);
    // hidden_prev_grad += reset_hidden_grad * batch_gate_grad[1]
    Tensor tmp;
    tmp.mutable_data<T>(mpc_hidden_prev_grad_t.dims(), context.GetPlace());
    mpc_operator->mul(&mpc_reset_hidden_prev_grad_t, &mpc_splitted_gate_grad_t[1], &tmp);
    mpc_operator->add(&mpc_hidden_prev_grad_t, &tmp, &mpc_hidden_prev_grad_t);
    // batch_gate_grad[0] = sigmoid_grad(batch_gate_grad[0], batch_gate[0])
    ComputeSigmoidGrad<T>(context, mpc_splitted_gate_grad_t[0],
                          mpc_splitted_gate_t[0], mpc_splitted_gate_grad_t[0]);
    // batch_gate_grad[1] = sigmoid_grad(batch_gate_grad[1], batch_gate[1])
    ComputeSigmoidGrad<T>(context, mpc_splitted_gate_grad_t[1],
                          mpc_splitted_gate_t[1], mpc_splitted_gate_grad_t[1]);
}

template<typename T>
inline void ComputeSigmoidGrad(const framework::ExecutionContext& context,
                               Tensor& dy, Tensor& y, Tensor& dx) {
    // dx = dy * (1.0 - y * y);
    PADDLE_ENFORCE_NOT_NULL(mpc::MpcInstance::mpc_protocol,
                            "Protocol %s is not yet created in MPC Protocol.");
    auto mpc_operator = mpc::MpcInstance::mpc_instance()->mpc_protocol()->mpc_operators();
    Tensor tmp;
    tmp.mutable_data<T>(dx.dims(), context.GetPlace());
    mpc_operator->mul(&y, &y, &tmp);
    mpc_operator->mul(&dy, &tmp, &tmp);
    mpc_operator->sub(&dy, &tmp, &dx);
}

template <typename DeviceContext, typename T>
class MpcGRUGradKernel : public MpcOpKernel<T> {
public:
    void BatchCompute(const framework::ExecutionContext& context) const {
        bool origin_mode = context.Attr<bool>("origin_mode");
        auto* h0 = context.Input<Tensor>("H0");
        auto* weight = context.Input<Tensor>("Weight");
        const T* weight_data = weight->data<T>();
        auto* batch_gate = context.Input<LoDTensor>("BatchGate");
        auto* batch_reset_hidden_prev =
            context.Input<LoDTensor>("BatchResetHiddenPrev");
        auto* batch_hidden = context.Input<LoDTensor>("BatchHidden");
        auto* hidden = context.Input<LoDTensor>("Hidden");
        auto* hidden_grad =
            context.Input<LoDTensor>(framework::GradVarName("Hidden"));
        auto* input_grad =
            context.Output<LoDTensor>(framework::GradVarName("Input"));
        auto* h0_grad = context.Output<Tensor>(framework::GradVarName("H0"));
        auto* weight_grad =
            context.Output<Tensor>(framework::GradVarName("Weight"));
        auto* bias_grad = context.Output<Tensor>(framework::GradVarName("Bias"));

        auto gate_dims = batch_gate->dims();
        auto hidden_dims = hidden->dims();
        auto gate_lod = batch_gate->lod();
        const auto& place = context.GetPlace();
        bool has_hidden_prev = false;
        bool has_hidden_prev_grad = false;
        bool has_weight_grad = false;

        math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
        LoDTensor batch_hidden_grad, batch_gate_grad, batch_reset_hidden_prev_grad;
        batch_hidden_grad.mutable_data<T>(hidden_dims, context.GetPlace());
        batch_gate_grad.mutable_data<T>(gate_dims, context.GetPlace());
        batch_reset_hidden_prev_grad.mutable_data<T>(hidden_dims,
                context.GetPlace());
        math::SetConstant<DeviceContext, T> zero;
        auto& dev_ctx = context.template device_context<DeviceContext>();
        zero(dev_ctx, &batch_hidden_grad, static_cast<T>(0));
        zero(dev_ctx, &batch_gate_grad, static_cast<T>(0));
        zero(dev_ctx, &batch_reset_hidden_prev_grad, static_cast<T>(0));

        Tensor ordered_h0, ordered_h0_grad;

        framework::Vector<size_t> order(gate_lod[2]);

        if (h0) {
            // Reorder mpc h0
            ordered_h0.mutable_data<T>(h0->dims(), place);
            for (int i = 0; i < 2; ++i) {
                Tensor h0_s;
                SliceAndReshape(h0, h0_s, i);
                Tensor ordered_h0_s;
                SliceAndReshape(&ordered_h0, ordered_h0_s, i);
                ReorderInitState<DeviceContext, T>(dev_ctx, h0_s, order, &ordered_h0_s,
                                                   true);
            }
        }
        if (h0_grad) {
            ordered_h0_grad.mutable_data<T>(h0_grad->dims(), context.GetPlace());
            zero(context.template device_context<DeviceContext>(), &ordered_h0_grad,
                 static_cast<T>(0));
        }

        bool is_reverse = context.Attr<bool>("is_reverse");
        for (int i = 0; i < 2; ++i) {
            // mpc LoDTensor to mpc batch
            Tensor batch_hidden_grad_s;
            SliceAndReshape(&batch_hidden_grad, batch_hidden_grad_s, i);
            Tensor hidden_grad_s;
            SliceAndReshape(hidden_grad, hidden_grad_s, i);
            LoDTensor lod_batch_hidden_grad_s;
            LoDTensor lod_hidden_grad_s;
            lod_batch_hidden_grad_s.ShareBufferWith(batch_hidden_grad_s);
            lod_batch_hidden_grad_s.mutable_data<T>(batch_hidden_grad_s.dims(), place);
            lod_hidden_grad_s.ShareBufferWith(hidden_grad_s);
            lod_hidden_grad_s.mutable_data<T>(hidden_grad_s.dims(), place);
            lod_hidden_grad_s.set_lod(gate_lod);
            lod_batch_hidden_grad_s.set_lod(gate_lod);
            to_batch(dev_ctx, lod_hidden_grad_s, &lod_batch_hidden_grad_s, false, is_reverse);
        }
        if (weight_grad) {
            T* gate_weight_grad =
                weight_grad->mutable_data<T>(context.GetPlace());
            zero(dev_ctx, weight_grad, static_cast<T>(0));
            has_weight_grad = true;
        }
        // split weights
        std::vector<Tensor> mpc_splitted_weights_t;
        SplitWeight<DeviceContext, T>(context, mpc_splitted_weights_t, *weight);

        auto batch_starts = gate_lod[0];
        size_t num_batch = batch_starts.size() - 1;
        for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
            int bstart = static_cast<int>(batch_starts[n]);
            int bend = static_cast<int>(batch_starts[n + 1]);
            int cur_batch_size = bend - bstart;
            int bstart_pre = static_cast<int>(batch_starts[n - 1]);

            // Split mpc tensors
            Tensor mpc_hidden_grad_t;
            Tensor mpc_hidden_prev_t;
            Tensor mpc_hidden_prev_grad_t;
            Tensor mpc_reset_hidden_prev_t;
            Tensor mpc_reset_hidden_prev_grad_t;
            std::vector<Tensor> splitted_batch_gate_t;
            std::vector<Tensor> mpc_splitted_gate_t;
            std::vector<Tensor> splitted_batch_gate_grad_t;
            std::vector<Tensor> mpc_splitted_gate_grad_t;
            std::vector<Tensor> mpc_splitted_weights_grad_t;

            if (weight_grad) {
                SplitWeight<DeviceContext, T>(context, mpc_splitted_weights_grad_t, *weight_grad);
            }
            ToMpcBatchTensor<DeviceContext, T>(context, mpc_hidden_grad_t, batch_hidden_grad, bstart, bend);
            ToMpcBatchTensor<DeviceContext, T>(context, mpc_reset_hidden_prev_t, *batch_reset_hidden_prev, bstart, bend);
            ToMpcBatchTensor<DeviceContext, T>(context, mpc_reset_hidden_prev_grad_t,
                                               batch_reset_hidden_prev_grad, bstart, bend);

            Split3Dim<DeviceContext, T>(context, splitted_batch_gate_grad_t, batch_gate_grad);
            Split3Dim<DeviceContext, T>(context, splitted_batch_gate_t, *batch_gate);
            for (int i = 0; i < 3; ++i) {
                ToMpcBatchTensor<DeviceContext, T>(context, mpc_splitted_gate_grad_t[i],
                                                   splitted_batch_gate_grad_t[i], bstart, bend);
                ToMpcBatchTensor<DeviceContext, T>(context, mpc_splitted_gate_t[i],
                                                   splitted_batch_gate_t[i], bstart, bend);
            }
            if (n == 0) {
                if (h0) {
                    // hidden_prev_t = ordered_h0
                    mpc_hidden_prev_t.mutable_data<T>(
                                             ordered_h0.dims(), place);
                    framework::TensorCopy(ordered_h0, place, &mpc_hidden_prev_t);
                    has_hidden_prev = true;
                    if (h0_grad) {
                        // hidden_prev_grad_t = ordered_h0_grad
                        mpc_hidden_prev_grad_t.mutable_data<T>(
                                                      ordered_h0_grad.dims(), place);
                        framework::TensorCopy(ordered_h0_grad, place, &mpc_hidden_prev_grad_t);
                        has_hidden_prev_grad = true;
                    }
                }
            } else {
                ToMpcBatchTensor<DeviceContext, T>(context, mpc_hidden_prev_t, *batch_hidden, bstart_pre, bstart);
                ToMpcBatchTensor<DeviceContext, T>(context, mpc_hidden_prev_grad_t, batch_hidden_grad, bstart_pre, bstart);

            }
            // compute GRUUnitGrad
            GRUUnitGradCompute<DeviceContext, T>(context,
                                                 mpc_splitted_gate_t, mpc_splitted_gate_grad_t,
                                                 mpc_hidden_prev_t, mpc_hidden_prev_grad_t,
                                                 mpc_splitted_weights_t, mpc_splitted_weights_grad_t,
                                                 mpc_reset_hidden_prev_t, mpc_reset_hidden_prev_grad_t,
                                                 mpc_hidden_grad_t, origin_mode, has_hidden_prev,
                                                 has_hidden_prev_grad, has_weight_grad);
            // cancat mpc tensor to gru_grad output variables
            if (weight_grad) {
                ConcatWeight<DeviceContext, T>(context, weight_grad, mpc_splitted_weights_grad_t);
            }
            Tensor mpc_batch_gate_grad_t;
            Concat3Dim<DeviceContext, T>(context, &mpc_batch_gate_grad_t, mpc_splitted_gate_grad_t);
            ConcatBatchOne<DeviceContext, T>(context, &batch_gate_grad, mpc_batch_gate_grad_t, bstart, bend);
            ConcatBatchOne<DeviceContext, T>(context, &batch_hidden_grad, mpc_hidden_prev_grad_t, bstart_pre, bstart);
            ConcatBatchOne<DeviceContext, T>(context, &batch_reset_hidden_prev_grad, mpc_reset_hidden_prev_grad_t, bstart, bend);
        }
        if (input_grad) {
            // batch to lodTensor for mpc input_grad
            input_grad->mutable_data<T>(context.GetPlace());
            math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
            batch_gate_grad.set_lod(gate_lod);
            for (int i = 0; i < 2; ++i) {
                Tensor batch_gate_grad_s;
                SliceAndReshape(&batch_gate_grad, batch_gate_grad_s, i);
                Tensor input_grad_s;
                SliceAndReshape(input_grad, input_grad_s, i);

                LoDTensor lod_batch_gate_grad_s;
                LoDTensor lod_input_grad_s;
                lod_batch_gate_grad_s.ShareBufferWith(batch_gate_grad_s);
                lod_batch_gate_grad_s.mutable_data<T>(batch_gate_grad_s.dims(), place);
                lod_batch_gate_grad_s.set_lod(gate_lod);
                lod_input_grad_s.ShareBufferWith(input_grad_s);
                lod_input_grad_s.mutable_data<T>(input_grad_s.dims(), place);
                to_seq(dev_ctx, lod_batch_gate_grad_s, &lod_input_grad_s);
            }
        }
        if (bias_grad) {
            // col_sum mpc bias_grad
            bias_grad->mutable_data<T>(context.GetPlace());
            math::ColwiseSum<DeviceContext, T> col_sum;
            for (int i = 0; i < 2; ++i) {
                Tensor batch_gate_grad_s;
                SliceAndReshape(&batch_gate_grad, batch_gate_grad_s, i);
                Tensor bias_grad_s;
                SliceAndReshape(bias_grad, bias_grad_s, i);
                col_sum(dev_ctx, batch_gate_grad_s, &bias_grad_s);
            }
        }
        if (h0 && h0_grad) {
            // Reorder mpc h0_grad
            for (int i = 0; i < 2; ++i) {
                Tensor ordered_h0_grad_s;
                SliceAndReshape(&ordered_h0_grad, ordered_h0_grad_s, i);
                Tensor h0_grad_s;
                SliceAndReshape(h0_grad, h0_grad_s, i);
                ReorderInitState<DeviceContext, T>(dev_ctx, ordered_h0_grad_s, order,
                                                   &h0_grad_s, false);
            }
        }
    }

    void ComputeImpl(const framework::ExecutionContext& context) const override {
        BatchCompute(context);
    }
};

}  // namespace operators
}  // namespace paddle


