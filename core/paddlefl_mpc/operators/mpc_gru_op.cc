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

#include "mpc_gru_op.h"

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "mpc_op.h"

#include <memory>
#include <string>
#include "core/paddlefl_mpc/operators/math/math_function.h"

namespace paddle
{
namespace operators
{

using framework::DDim;
using framework::Tensor;
using framework::LoD;

class MpcGRUOp : public framework::OperatorWithKernel
{
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext *ctx) const override
    {
        PADDLE_ENFORCE(ctx->HasInput("Input"),
                       "Input(%s) of MpcGRUOp should not be null.", "Input");
        PADDLE_ENFORCE(ctx->HasInput("Weight"),
                       "Input(%s) of MpcGRUOp should not be null.", "Weight");
        PADDLE_ENFORCE(ctx->HasOutput("BatchGate"),
                       "Output(%s) of MpcGRUOp should not be null.", "BatchGate");
        PADDLE_ENFORCE(ctx->HasOutput("BatchResetHiddenPrev"),
                       "Output(%s) of MpcGRUOp should not be null.",
                       "BatchResetHiddenPrev");
        PADDLE_ENFORCE(ctx->HasOutput("BatchHidden"),
                       "Output(%s) of MpcGRUOp should not be null.", "BatchHidden");
        PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                       "Output(%s) of MpcGRUOp should not be null.", "Hidden");
        auto input_dims_trans = ctx->GetInputDim("Input");
        auto input_dims = framework::make_ddim({input_dims_trans[1],
                                                input_dims_trans[0], input_dims_trans[2]});
        auto weight_dims = ctx->GetInputDim("Weight");
        int input_size = input_dims[2];
        int frame_size = weight_dims[1];
        if (ctx->IsRuntime())
        {
            PADDLE_ENFORCE_EQ(
                input_size, frame_size * 3,
                "The input_size must be 3 times of frame_size in MpcGRUOp.");
        }
        PADDLE_ENFORCE_EQ(
            weight_dims[2], frame_size * 3,
            "The shape of mpc Weight matrix must be [frame_size, frame_size * 3].");
        if (ctx->HasInput("H0"))
        {
            auto h0_dims = ctx->GetInputDim("H0");
            PADDLE_ENFORCE_EQ(h0_dims[2], frame_size,
                              "The width of H0 must be equal to frame_size.");
        }
        if (ctx->HasInput("Bias"))
        {
            auto bias_dims = ctx->GetInputDim("Bias");
            int bias_height = bias_dims[1];
            int bias_width = bias_dims[2];
            PADDLE_ENFORCE_EQ(bias_height, 1,
                              "The shape of Bias must be [1, frame_size * 3].");
            PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                              "The shape of Bias must be [1, frame_size * 3].");
        }
        ctx->SetOutputDim("BatchGate", input_dims);
        ctx->SetOutputDim("BatchResetHiddenPrev", {2, input_dims[1], frame_size});
        ctx->SetOutputDim("BatchHidden", {2, input_dims[1], frame_size});
        ctx->SetOutputDim("Hidden", {2, input_dims[1], frame_size});
        ctx->ShareLoD("Input", "Hidden");
    }
};

class MpcGRUOpMaker : public framework::OpProtoAndCheckerMaker
{
public:
    void Make() override
    {
        AddInput("Input",
                 "(LoDTensor) The first input is a LodTensor, which supports "
                 "variable-time length input sequence. The underlying tensor in "
                 "this LoDTenosr is a matrix with shape (T x 2 x 3D), where, T is the "
                 "total time steps in this mini-batch, D is the hidden size."
                 "Note: before call this OP, "
                 "Yout must transpose input shape of mini-batch dim to first dim,"
                 "that is, (2, T, 3D) is transpose to (T, 2, 3D), "
                 "so that its lod information of shares can be set correctly");
        AddInput("H0",
                 "(Tensor, optional) The initial hidden state is an optional "
                 "input. This is a tensor with shape (2 x N x D), where N is the "
                 "batch size, D is the hidden size.")
        .AsDispensable();
        AddInput(
            "Weight",
            "(Tensor) The learnable hidden-hidden weight matrix with shape "
            "(2 x D x 3D), where D is the hidden size. The elements continuous in "
            "memory can be divided into two parts. The first part are weights of "
            "the update gate and reset gate with shape (2 x D x 2D), and the second "
            "part are weights of output candidate with shape (2 x D x D).");
        AddInput("Bias",
                 "(Tensor, optional) Bias vector with shape (2 x 1 x 3D) concating "
                 "bias of the update gate, reset gate and output candidate.")
        .AsDispensable();
        AddOutput("BatchGate",
                  "(LoDTensor) To compute with batches, sequence data will be "
                  "reorganized into several successive batches each containing "
                  "data from the same time step. The LoDTensor BatchGate contains "
                  "the update gate, reset gate and output candidate values "
                  "organized in batches. The LoD size is 2. The first LoD contains "
                  "the batch offsets and the second LoD contains the indexes in "
                  "the raw sequence data.")
        .AsIntermediate();
        AddOutput(
            "BatchResetHiddenPrev",
            "(LoDTensor) The reset hidden state LoDTensor organized in batches. "
            "This LoDTensor is a matrix with shape (2 x T x D) and has the same LoD "
            "with `BatchGate`.")
        .AsIntermediate();
        AddOutput(
            "BatchHidden",
            "(LoDTensor) The hidden state LoDTensor organized in batches.  "
            "This LoDTensor is a matrix with shape (2 x T x D) and has the same LoD "
            "with `BatchGate`.")
        .AsIntermediate();
        AddOutput(
            "Hidden",
            "(LoDTensor) the hidden state LoDTensor organized in sequences. "
            "This LoDTensor is a matrix with shape (2 x T x D) and has the same LoD "
            "with `BatchGate`.");
        AddAttr<std::string>("activation",
                             "(string, default tanh) "
                             "The activation type used for output candidate {h}_t.")
        .SetDefault("relu");
        AddAttr<std::string>(
            "gate_activation",
            "(string, default sigmoid) "
            "The activation type used in update gate and reset gate.")
        .SetDefault("sigmoid");
        AddAttr<bool>("is_reverse",
                      "(bool, default: False) "
                      "whether to compute reversed GRU.")
        .SetDefault(false);
        AddAttr<bool>("origin_mode",
                      "bool"
                      "use origin mode in article https://arxiv.org/abs/1412.3555")
        .SetDefault(false);
        AddComment(R"DOC(
GRU Operator implements part calculations of the complete GRU as following:

$$
update\_gate: u_t = actGate(xu_t + W_u * h_{t-1} + b_u) \\
reset\_gate: r_t = actGate(xr_t + W_r * h_{t-1} + b_r)  \\
output\_candidate: {h}_t = actNode(xc_t + W_c * dot(r_t, h_{t-1}) + b_c) \\
output: h_t = dot((1 - u_t), h_{t-1}) + dot(u_t, {h}_t)
$$

@note To implement the complete GRU, fully-connected operator must be used
before to feed xu, xr and xc as the Input of GRU operator.
)DOC");
  }
};

class MpcGRUGradOp : public framework::OperatorWithKernel
{
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override
  {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(%s) of MpcGRUGradOp should not be null.", "Input");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(%s) of MpcGRUGradOp should not be null.", "Weight");
    PADDLE_ENFORCE(ctx->HasInput("BatchGate"),
                   "Input(%s) of MpcGRUGradOp should not be null.", "BatchGate");
    PADDLE_ENFORCE(ctx->HasInput("BatchResetHiddenPrev"),
                   "Input(%s) of MpcGRUGradOp should not be null.",
                   "BatchResetHiddenPrev");
    PADDLE_ENFORCE(ctx->HasInput("BatchHidden"),
                   "Input(%s) of MpcGRUOp should not be null.", "BatchHidden");
    PADDLE_ENFORCE(ctx->HasInput("Hidden"),
                   "Input(%s) of MpcGRUGradOp should not be null.", "Hidden");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Hidden")),
                   "Input(%s@GRAD) of MpcGRUGradOp should not be null.", "Hidden");
    auto input_dims_trans = ctx->GetInputDim("Input");
    auto input_dims = framework::make_ddim({input_dims_trans[1],
                            input_dims_trans[0], input_dims_trans[2]});
    auto weight_dims = ctx->GetInputDim("Weight");
    int input_size = input_dims[2];
    int frame_size = weight_dims[1];
    int weight_height = weight_dims[1];
    int weight_width = weight_dims[2];
    PADDLE_ENFORCE_EQ(input_size, frame_size * 3,
                      "The input_size must be 3 times of frame_size in MpcGRUOp.");
    PADDLE_ENFORCE_EQ(
        weight_height, frame_size,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(
        weight_width, frame_size * 3,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    if (ctx->HasInput("H0"))
    {
      auto h0_dims = ctx->GetInputDim("H0");
      PADDLE_ENFORCE_EQ(h0_dims[2], frame_size,
                        "The width of H0 must be equal to frame_size.");
      auto h0_grad_name = framework::GradVarName("H0");
      if (ctx->HasOutput(h0_grad_name))
        ctx->SetOutputDim(h0_grad_name, h0_dims);
    }
    if (ctx->HasInput("Bias"))
    {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[1];
      int bias_width = bias_dims[2];
      PADDLE_ENFORCE_EQ(bias_height, 1,
                        "The shape of Bias must be [1, frame_size * 3].");
      PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                        "The shape of Bias must be [1, frame_size * 3].");
      auto bias_grad_name = framework::GradVarName("Bias");
      if (ctx->HasOutput(bias_grad_name))
        ctx->SetOutputDim(bias_grad_name, bias_dims);
    }
    auto input_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(input_grad_name))
      //transpose input's shape
      ctx->SetOutputDim(input_grad_name, input_dims);
    auto weight_grad_name = framework::GradVarName("Weight");
    if (ctx->HasOutput(weight_grad_name))
      ctx->SetOutputDim(weight_grad_name, weight_dims);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override
  {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Hidden")),
                                   ctx.device_context());
  }
};

template <typename T>
class MpcGRUCPUKernel : public MpcOpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    bool origin_mode = context.Attr<bool>("origin_mode");
    auto* input_trans = context.Input<LoDTensor>("Input");
    auto* h0 = context.Input<Tensor>("H0");
    auto* weight = context.Input<Tensor>("Weight");
    const T* weight_data = weight->data<T>();
    auto* bias = context.Input<Tensor>("Bias");
    auto* batch_gate = context.Output<LoDTensor>("BatchGate");
    batch_gate->mutable_data<T>(context.GetPlace());
    auto* batch_reset_hidden_prev =
        context.Output<LoDTensor>("BatchResetHiddenPrev");
    batch_reset_hidden_prev->mutable_data<T>(context.GetPlace());
    auto* batch_hidden = context.Output<LoDTensor>("BatchHidden");
    batch_hidden->mutable_data<T>(context.GetPlace());
    auto* hidden = context.Output<LoDTensor>("Hidden");
    hidden->mutable_data<T>(context.GetPlace());

    auto hidden_dims = hidden->dims();
    const auto place = context.GetPlace();

    bool is_reverse = context.Attr<bool>("is_reverse");

    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    // get input lod
    auto input_lod = input_trans->lod();
    LoD gate_lod;
    // transpose input to corrected mpc_input
    // (T, 2, 3D) to (2, T, 3D)
    math::Transpose<DeviceContext, T, 3> transpose;
    Tensor input;
    auto input_dim = input_trans->dims();
    auto in_dim = framework::make_ddim({input_dim[1], input_dim[0], input_dim[2]});
    input.mutable_data<T>(
        in_dim,
        context.GetPlace());
    transpose(dev_ctx, *input_trans, &input, {1, 0, 2});

    for (int i = 0; i < 2; ++i) {
        // mpc LoDTensor to Batch
        Tensor input_s;
        Tensor batch_gate_s;
        SliceAndReshape(&input, input_s, i);
        SliceAndReshape(batch_gate, batch_gate_s, i);
        LoDTensor lod_input_s;
        LoDTensor lod_batch_gate_s;
        lod_input_s.ShareBufferWith(input_s);
        lod_input_s.mutable_data<T>(input_s.dims(), place);
        lod_batch_gate_s.ShareBufferWith(batch_gate_s);
        lod_batch_gate_s.mutable_data<T>(batch_gate_s.dims(), place);
        lod_input_s.set_lod(input_lod);
        to_batch(dev_ctx, lod_input_s, &lod_batch_gate_s, true, is_reverse);
        gate_lod = lod_batch_gate_s.lod();
    }

    if (bias) {
        // add mpc bias
        math::RowwiseAdd<DeviceContext, T> add_bias;
        for (int i = 0; i < 2; ++i) {
            Tensor batch_gate_s;
            Tensor bias_s;
            SliceAndReshape(batch_gate, batch_gate_s, i);
            SliceAndReshape(bias, bias_s, i);
            add_bias(dev_ctx, batch_gate_s, bias_s, &batch_gate_s);
        }
    }
    // split mpc weight from shape (2, D, 3D) to 3 * (2, D, D)
    std::vector<Tensor> mpc_splitted_weights_t;
    //Split3Dim<DeviceContext, T>(context, &mpc_splitted_weights_t, *weight);
    SplitWeight<DeviceContext, T>(context, mpc_splitted_weights_t, *weight);

    Tensor ordered_h0;
    framework::Vector<size_t> order((gate_lod)[2]);
    Tensor mpc_hidden_prev_t;
    bool has_hidden_prev = false;

    if (h0) {
      // reordered h0 based on lod
      ordered_h0.mutable_data<T>(h0->dims(), place);
      for (int i = 0; i < 2; ++i) {
          Tensor h0_s;
          Tensor ordered_h0_s;
          SliceAndReshape(h0, h0_s, i);
          SliceAndReshape(&ordered_h0, ordered_h0_s, i);
          ReorderInitState<DeviceContext, T>(
                context.template device_context<DeviceContext>(), h0_s, order,
                &ordered_h0_s, true);
      }
      // copy ordered_h0 to mpc_hidden_prev_t
      mpc_hidden_prev_t = ordered_h0;
      has_hidden_prev = true;
    }
    auto batch_starts = (gate_lod)[0];
    size_t seq_len = batch_starts.size() - 1;

    std::vector<Tensor> mpc_gate_t_list;
    std::vector<Tensor> mpc_reset_hidden_prev_t_list;
    std::vector<Tensor> mpc_hidden_t_list;
    // compute gru
    for (size_t n = 0; n < seq_len; n++) {
        int bstart = static_cast<int>(batch_starts[n]);
        int bend = static_cast<int>(batch_starts[n + 1]);
        int cur_batch_size = bend - bstart;

        std::vector<Tensor> mpc_splitted_gate_t;
        Tensor mpc_batch_gate_t;
        Tensor mpc_reset_hidden_prev_t;
        Tensor mpc_hidden_t;

        ToMpcBatchTensor<DeviceContext, T>(context, mpc_batch_gate_t, *batch_gate, bstart, bend);
        Split3Dim<DeviceContext, T>(context, mpc_splitted_gate_t, mpc_batch_gate_t);
        ToMpcBatchTensor<DeviceContext, T>(context, mpc_reset_hidden_prev_t, *batch_reset_hidden_prev, bstart, bend);
        ToMpcBatchTensor<DeviceContext, T>(context, mpc_hidden_t, *batch_hidden, bstart, bend);

        ComputGRUUint<DeviceContext, T>(context, mpc_splitted_gate_t, mpc_splitted_weights_t, mpc_reset_hidden_prev_t,
                    mpc_hidden_t, mpc_hidden_prev_t, origin_mode, has_hidden_prev);

        Tensor mpc_gate_t;
        Concat3Dim<DeviceContext, T>(context, &mpc_gate_t, mpc_splitted_gate_t);
        //mpc_hidden_prev_t = mpc_hidden_t;
        mpc_hidden_prev_t.mutable_data<T>(mpc_hidden_t.dims(), place);
        framework::TensorCopy(mpc_hidden_t, context.GetPlace(), &mpc_hidden_prev_t);
        mpc_gate_t_list.emplace_back(mpc_gate_t);
        mpc_reset_hidden_prev_t_list.emplace_back(mpc_reset_hidden_prev_t);
        mpc_hidden_t_list.emplace_back(mpc_hidden_t);
    }
    // Concat output variables
    ConcatBatchAll<DeviceContext, T>(context, batch_gate, mpc_gate_t_list);
    ConcatBatchAll<DeviceContext, T>(context, batch_reset_hidden_prev, mpc_reset_hidden_prev_t_list);
    ConcatBatchAll<DeviceContext, T>(context, batch_hidden, mpc_hidden_t_list);
    // mpc batch tensor to mpc LoDTensor
    for (int i = 0; i < 2; ++i)
    {
        Tensor batch_hidden_s;
        SliceAndReshape(batch_hidden, batch_hidden_s, i);
        Tensor hidden_s;
        SliceAndReshape(hidden, hidden_s, i);
        LoDTensor lod_batch_hidden_s;
        LoDTensor lod_hidden_s;

        lod_batch_hidden_s.ShareBufferWith(batch_hidden_s);
        lod_batch_hidden_s.mutable_data<T>(batch_hidden_s.dims(), place);
        lod_hidden_s.ShareBufferWith(hidden_s);
        lod_hidden_s.mutable_data<T>(hidden_s.dims(), place);
        math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
        lod_batch_hidden_s.set_lod(gate_lod);
        lod_hidden_s.set_lod(gate_lod);
        to_seq(dev_ctx, lod_batch_hidden_s, &lod_hidden_s);
    }
    // set batch_gate_lod for grad op
    batch_gate->set_lod(gate_lod);
  }

  void ComputeImpl(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

template <typename T>
class MpcGRUGradOpMaker : public framework::SingleGradOpMaker<T>
{
public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
  void Apply(GradOpPtr<T> grad_op) const override
  {
    grad_op->SetType("mpc_gru_grad");
    grad_op->SetInput("Input", this->Input("Input"));
    grad_op->SetInput("H0", this->Input("H0"));
    grad_op->SetInput("Bias", this->Input("Bias"));
    grad_op->SetInput("Weight", this->Input("Weight"));

    grad_op->SetInput("BatchGate", this->Output("BatchGate"));
    grad_op->SetInput("BatchResetHiddenPrev",
                      this->Output("BatchResetHiddenPrev"));
    grad_op->SetInput("BatchHidden", this->Output("BatchHidden"));
    grad_op->SetInput("Hidden", this->Output("Hidden"));

    grad_op->SetInput(framework::GradVarName("Hidden"),
                      this->OutputGrad("Hidden"));

    grad_op->SetOutput(framework::GradVarName("H0"), this->InputGrad("H0"));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetOutput(framework::GradVarName("Weight"),
                       this->InputGrad("Weight"));
    grad_op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));

    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(MpcGRUGradOpNoNeedBufferVarInference, "Input",
                                    "Bias");

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mpc_gru, ops::MpcGRUOp, ops::MpcGRUOpMaker,
                  ops::MpcGRUGradOpMaker<paddle::framework::OpDesc>,
                  ops::MpcGRUGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mpc_gru_grad, ops::MpcGRUGradOp,
                  ops::MpcGRUGradOpNoNeedBufferVarInference);
REGISTER_OP_CPU_KERNEL(mpc_gru, ops::MpcGRUCPUKernel<int64_t>);
REGISTER_OP_CPU_KERNEL(
    mpc_gru_grad, ops::MpcGRUGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
