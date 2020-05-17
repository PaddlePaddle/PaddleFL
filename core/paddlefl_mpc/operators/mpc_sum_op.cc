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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/framework/op_registry.h"
#include "mpc_sum_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MpcSumOp : public framework::OperatorWithKernel {
public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInputs("X"), true, 
            platform::errors::NotFound("Input(X) of MpcElementwiseAddOp should not be null."));

        PADDLE_ENFORCE_EQ(
            ctx->HasOutput("Out"), true,
            platform::errors::NotFound("Output(Out) of MulOp should not be null."));

        auto x_var_types = ctx->GetInputsVarType("X");
        auto x_dims = ctx->GetInputsDim("X");
        auto N = x_dims.size();
        PADDLE_ENFORCE_GT(
            N, 0,
            "ShapeError: The input tensor X's dimensions of SumOp "
            "should be larger than 0. But received X's dimensions %d, "
            "X's shape = [%s].",
            N, &x_dims);
        if (N == 1) {
            VLOG(3) << "Warning: SumOp have only one input, may waste memory";
        }
        
        framework::DDim in_dim({0});
        for (size_t i = 0; i < x_dims.size(); ++i) {
            auto& x_dim = x_dims[i];
            // x_dim.size() == 1 means the real dim of selected rows is [0]
            if (x_var_types[i] == framework::proto::VarType::SELECTED_ROWS &&
                x_dim.size() == 1) {
                continue;
            }
           if (framework::product(x_dim) == 0) {
               continue;
           }
           if (framework::product(in_dim) == 0) {
               in_dim = x_dim;
           } else {
               if (ctx->IsRuntime()) {
                   PADDLE_ENFORCE_EQ(
                       in_dim, x_dim,
                       "ShapeError: The input tensor X of SumOp must have same shape."
                       "But received X[0]'s shape = [%s], X[%d]'s shape = [%s].",
                       in_dim, i, x_dim);
               } else {
                   PADDLE_ENFORCE_EQ(
                   in_dim.size(), x_dim.size(),
                       "ShapeError: The input tensor X of SumOp must have same "
                       "dimensions. But received X[0]'s dimensions = %d, X[0]'s shape = "
                       "[%s], X[%d]'s dimensions = %d, X[%d]'s shape = [%s].",
                       in_dim.size(), in_dim, i, x_dim.size(), i, x_dim);
                    // if in_dim or x_dim has -1, not check equal
                    for (int j = 0; j < x_dim.size(); ++j) {
                        if (x_dim[j] == -1 || in_dim[j] == -1) {
                            continue;
                        }
                        PADDLE_ENFORCE_EQ(
                            in_dim[j], x_dim[j],
                            "ShapeError: The input tensor X of SumOp must have same shape "
                            "if not -1."
                            "But received X[0]'s shape = [%s], X[%d]'s shape = [%s].",
                            in_dim, i, x_dim);
                    }
               }
           }
        }

        ctx->SetOutputDim("Out", in_dim);
        ctx->ShareLoD("X", /*->*/ "Out");
    }

};

class MpcSumOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
      AddInput("X",
             "A Varaible list. The shape and data type of the list elements"
             "should be consistent. Variable can be multi-dimensional Tensor"
             "or LoDTensor, and data types can be: float32, float64, int32, "
             "int64.")
         .AsDuplicable();
      AddOutput("Out",
              "the sum of input :code:`x`. its shape and data types are "
              "consistent with :code:`x`.");
      AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
      AddComment(R"DOC(This OP is used to sum one or more Tensor or LoDTensor
                    of the input. If the input is LoDTensor, the output only
                    shares LoD information with the first input.)DOC");
    }
};


class MpcSumGradMaker : public framework::GradOpDescMakerBase {
public:
    using framework::GradOpDescMakerBase::GradOpDescMakerBase;

    std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
        auto x_grads = InputGrad("X", false);
        std::vector<std::unique_ptr<framework::OpDesc>> grad_ops;
        grad_ops.reserve(x_grads.size());
        auto og = OutputGrad("Out");
        std::transform(x_grads.begin(), x_grads.end(), std::back_inserter(grad_ops),
                      [&og](const std::string& x_grad) {
                        auto* grad_op = new framework::OpDesc();
                        grad_op->SetType("scale");
                        grad_op->SetInput("X", og);
                        grad_op->SetOutput("Out", {x_grad});
                        grad_op->SetAttr("scale", 1.0f);
                        return std::unique_ptr<framework::OpDesc>(grad_op);
                      });

        return grad_ops;
    }
};

DECLARE_INPLACE_OP_INFERER(MpcSumInplace, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

//REGISTER_OP_WITHOUT_GRADIENT(mpc_sum, ops::MpcSumOp, ops::MpcSumOpMaker);
REGISTER_OPERATOR(mpc_sum, ops::MpcSumOp, 
                  ops::MpcSumOpMaker, 
                  ops::MpcSumGradMaker, 
                  ops::MpcSumInplace);

REGISTER_OP_CPU_KERNEL(mpc_sum, ops::MpcSumKernel<paddle::platform::CPUDeviceContext, int64_t>);
