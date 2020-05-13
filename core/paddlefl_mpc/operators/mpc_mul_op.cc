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

#include "paddle/fluid/framework/op_registry.h"
#include "mpc_mul_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcMulOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::NotFound("Input(X) of Mpc MulOp should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("Y"), true,
            platform::errors::NotFound("Input(Y) of MpcMulOp should not be null."));
        PADDLE_ENFORCE_EQ(
            ctx->HasOutput("Out"), true,
            platform::errors::NotFound("Output(Out) of MpcMulOp should not be null."));

        auto x_dims = ctx->GetInputDim("X");
        auto y_dims = ctx->GetInputDim("Y");

        int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
        int y_num_col_dims = ctx->Attrs().Get<int>("y_num_col_dims");

        VLOG(3) << "mpc mul operator x.shape=" << x_dims << " y.shape=" << y_dims
                << " x_num_col_dims=" << x_num_col_dims
                << " y_num_col_dims=" << y_num_col_dims;

        PADDLE_ENFORCE_NE(framework::product(y_dims), 0,
                          platform::errors::PreconditionNotMet(
                              "The Input variable Y(%s) has not "
                              "been initialized. You may need to confirm "
                              "if you put exe.run(startup_program) "
                              "after optimizer.minimize function.",
                              ctx->Inputs("Y").front()));
        PADDLE_ENFORCE_GT(
            x_dims.size(), x_num_col_dims,
            platform::errors::InvalidArgument(
                "The input tensor X's dimensions of MpcMulOp "
                "should be larger than x_num_col_dims. But received X's "
                "dimensions = %d, X's shape = [%s], x_num_col_dims = %d.",
                x_dims.size(), x_dims, x_num_col_dims));
        PADDLE_ENFORCE_GT(
            y_dims.size(), y_num_col_dims,
            platform::errors::InvalidArgument(
                "The input tensor Y's dimensions of MpcMulOp "
                "should be larger than y_num_col_dims. But received Y's "
                "dimensions = %d, Y's shape = [%s], y_num_col_dims = %d.",
                y_dims.size(), y_dims, y_num_col_dims));

        int x_mat_width = 1;
        int y_mat_height = 1;
        for (size_t i = x_num_col_dims + 1; i < x_dims.size(); i++) {
            x_mat_width *= x_dims[i];
        }
        for (size_t i = 1; i <= y_num_col_dims; i++) {
            y_mat_height *= y_dims[i];
        }

        PADDLE_ENFORCE_EQ(
            x_mat_width, y_mat_height,
            platform::errors::InvalidArgument(
                "After flatten the input tensor X and Y to 2-D dimensions "
                "matrix X1 and Y1, the matrix X1's width must be equal with matrix "
                "Y1's height. But received X's shape = [%s], X1's "
                "width = %s; Y's shape = [%s], Y1's height = %s.",
                x_dims, x_mat_width, y_dims, y_mat_height));

        std::vector<int64_t> output_dims;
        output_dims.reserve(
            static_cast<size_t>(1 + x_num_col_dims + y_dims.size() - y_num_col_dims));

	for (int i = 0; i <= x_num_col_dims; ++i) { // i=0, batch_size (share id)
            output_dims.push_back(x_dims[i]);
        }

        for (int i = y_num_col_dims + 1; i < y_dims.size(); ++i) {
            output_dims.push_back(y_dims[i]);
        }

        ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
        ctx->ShareLoD("X", /*->*/ "Out");
    }
};

class MpcMulOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("X", "(Tensor), The first input tensor of mpc mul op.");
        AddInput("Y", "(Tensor), The second input tensor of mpc mul op.");
        AddOutput("Out", "(Tensor), The output tensor of mpc mul op.");
        AddAttr<bool>("use_mkldnn",
                    "(bool, default false) Only used in mkldnn kernel")
          .SetDefault(false);
        AddAttr<int>(
          "x_num_col_dims",
          R"DOC((int, default 1), The mul_op can take tensors with more than two
                dimensions as its inputs. If the input $X$ is a tensor with more
                than two dimensions, $X$ will be flattened into a two-dimensional
                matrix first. The flattening rule is: the first `num_col_dims`
                will be flattened to form the first dimension of the final matrix
                (the height of the matrix), and the rest `rank(X) - num_col_dims`
                dimensions are flattened to form the second dimension of the final
                matrix (the width of the matrix). As a result, height of the
                flattened matrix is equal to the product of $X$'s first
                `x_num_col_dims` dimensions' sizes, and width of the flattened
                matrix is equal to the product of $X$'s last `rank(x) - num_col_dims`
                dimensions' size. For example, suppose $X$ is a 6-dimensional
                tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
                Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
                [24, 30].
          )DOC")
          .SetDefault(1)
          .EqualGreaterThan(1);
        AddAttr<int>(
          "y_num_col_dims",
          R"DOC((int, default 1), The mul_op can take tensors with more than two,
                dimensions as its inputs. If the input $Y$ is a tensor with more
                than two dimensions, $Y$ will be flattened into a two-dimensional
                matrix first. The attribute `y_num_col_dims` determines how $Y$ is
                flattened. See comments of `x_num_col_dims` for more details.
          )DOC")
          .SetDefault(1)
          .EqualGreaterThan(1);
        AddAttr<float>(
          "scale_x",
          "scale_x to be used for int8 mul input data x. scale_x has the"
          "same purpose as scale_in in OPs that support quantization."
          "Only to be used with MKL-DNN INT8")
          .SetDefault(1.0f);
        AddAttr<std::vector<float>>(
          "scale_y",
          "scale_y to be used for int8 mul input data y. scale_y has the"
          "same purpose as scale_weights in OPs that support quantization."
          "Only to be used with MKL-DNN INT8")
          .SetDefault({1.0f});
        AddAttr<float>("scale_out",
                    "scale_out to be used for int8 output data."
                    "Only used with MKL-DNN INT8")
          .SetDefault(1.0f);
        AddAttr<bool>(
          "force_fp32_output",
          "(bool, default false) Force quantize kernel output FP32, only "
          "used in quantized MKL-DNN.")
          .SetDefault(false);
          AddComment(R"DOC(
MPC mul Operator.
)DOC");
    }
};

class MpcMulOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
protected:
    std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
            const override {
        static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
        return m;
    }
};

class MpcMulGradOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;
    using Tensor = framework::Tensor;

    void InferShape(framework::InferShapeContext *ctx) const override {
        auto out_grad_name = framework::GradVarName("Out");
        PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) should not be null.");
        PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true, "Input(Y) should not be null.");
        PADDLE_ENFORCE_EQ(ctx->HasInput(out_grad_name), true,
                          "Input(Out@GRAD) should not be null.");
        auto x_grad_name = framework::GradVarName("X");
        auto y_grad_name = framework::GradVarName("Y");

        auto x_dims = ctx->GetInputDim("X");
        auto y_dims = ctx->GetInputDim("Y");

        if (ctx->HasOutput(x_grad_name)) {
            ctx->SetOutputDim(x_grad_name, x_dims);
        }
        if (ctx->HasOutput(y_grad_name)) {
            ctx->SetOutputDim(y_grad_name, y_dims);
        }
    }
};

template <typename T>
class MpcMulOpGradMaker : public framework::SingleGradOpMaker<T> {
public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
    void Apply(GradOpPtr<T> grad) const override {
        grad->SetType("mpc_mul_grad");
        grad->SetInput("X", this->Input("X"));
        grad->SetInput("Y", this->Input("Y"));
        grad->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
        grad->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
        grad->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
        grad->SetAttrMap(this->Attrs());
    }
};

}  // namespace operators
}  // namespace paddle


namespace ops = paddle::operators;
REGISTER_OPERATOR(mpc_mul, ops::MpcMulOp,
                 ops::MpcMulOpMaker,
                 ops::MpcMulOpInferVarType,
                 ops::MpcMulOpGradMaker<paddle::framework::OpDesc>);

REGISTER_OPERATOR(mpc_mul_grad, ops::MpcMulGradOp);

REGISTER_OP_CPU_KERNEL(
    mpc_mul,
    ops::MpcMulKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(
    mpc_mul_grad,
    ops::MpcMulGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
