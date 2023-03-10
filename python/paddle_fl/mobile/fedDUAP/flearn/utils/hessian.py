"""
Hessian
"""
import paddle
import math

import torch
from torch.autograd import Variable
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([paddle.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s ** 0.5
    s = s.item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if param.grad is None:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = paddle.autograd.grad(gradsH,
                              params,
                              grad_outputs = v,
                              only_inputs = True,
                              retain_graph = True,
                              create_graph = True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha = -group_product(w, v))
    return normalization(w)


class Hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data is not None and dataloader is None) or (data is None and
                                                         dataloader is not None)

        self.model = model
        self.model.eval()  # make model is in evaluation model
        self.criterion = criterion
        self.optimizer = paddle.optimizer.SGD(learning_rate = 0.1, parameters = model.parameters())

        if data is not None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph = True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                paddle.randint(low = 0, high = 2, shape = p.shape)
                for p in self.params
            ]
            # generate Rademacher random variables
            for i, v_i in enumerate(v):
                v_i = v_i.astype(paddle.float32)
                v_i[v_i == 0] = -1
                v[i] = v_i
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                self.optimizer.clear_grad()
                w_prime = [paddle.zeros(p.shape) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = paddle.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [paddle.randn(p.shape) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = paddle.zeros([iter, iter])
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]

            paddle.device.set_device("cpu")
            a_, b_ = paddle.linalg.eig(T)

            eigen_list = a_.astype(paddle.float32)
            weight_list = b_[0, :].astype(paddle.float32) ** 2
            eigen_list_full.append(list(eigen_list.numpy()))
            weight_list_full.append(list(weight_list.numpy()))

        return eigen_list_full, weight_list_full
