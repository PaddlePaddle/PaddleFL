import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DenseLinear(nn.Layer):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, use_bias=True, use_mask=False, **kwargs):
        super(DenseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = paddle.create_parameter(shape=[in_features, out_features], dtype='float32')
        if use_bias:
            self.bias = paddle.create_parameter(shape=[out_features], dtype='float32')
        else:
            self.bias = None

        self.use_mask = use_mask
        self.mask = paddle.ones_like(self.weight)
        self.use_bias = use_bias
        self.bias_mask = paddle.ones_like(self.bias)

        self.save_weight_data = None
        self.save_weight_grad = None

        self.save_bias_data = None
        self.save_bias_grad = None

        self.save_mask = None
        self.save_bias_mask = None

    def forward(self, inp: paddle.Tensor):
        """
        前向传播前，将对应 mask 位置的权重设置为 0
        :param inp:
        :return:
        """
        masked_weight = self.weight * self.mask if self.use_mask else self.weight
        masked_bias = self.bias

        if self.use_bias and self.bias is not None and self.bias_mask is not None:
            masked_bias = masked_bias * self.bias_mask
        # return nn.functional.linear(inp, masked_weight, masked_bias)
        return F.linear(inp, masked_weight, masked_bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def prune_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weight_val = self.weight[self.mask == 1.]
        sorted_abs_weight = weight_val.abs().sort()[0]
        thr = sorted_abs_weight[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def retain_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def retain_by_rank(self, rank):
        weights_val = self.weight[self.mask == 1.]
        sorted_abs_weights = weights_val.abs().sort(descending=True)[0]
        thr = sorted_abs_weights[rank]
        self.retain_by_threshold(thr)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = paddle.rand(self.mask.shape)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = rand_val.sort()[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    def recovery_mask(self):
        """
        恢复 mask
        :return:
        """
        self.mask = paddle.ones_like(self.weight)
        self.bias_mask = paddle.ones_like(self.bias)

    def move_data(self):
        paddle.device.set_device('gpu')

    def to(self, *args, **kwargs):
        paddle.device.set_device('gpu')

    def structured_by_rank(self, rank, rate, device: paddle.device):
        f, _ = self.weight.shape
        pruned_num = int(rate * f)
        ind = paddle.argsort(rank)[pruned_num:].to(device)  # preserved filter id
        ind = paddle.sort(ind)

        self.ind = ind

        self.prune_out_channels(ind)

    def save_layer_info(self):
        """
        保存当前层的权重以及 mask
        :return:
        """
        self.save_weight_data = self.weight.clone()
        self.save_weight_grad = self.weight.grad.clone() if self.weight.grad is not None else None

        self.save_bias_data = self.bias.clone() if self.bias is not None else None
        self.save_bias_grad = self.bias.grad.clone() if self.bias is not None and self.bias.grad is not None else None

        self.save_mask = self.mask.clone()
        self.save_bias_mask = self.bias_mask.clone()

    def recovery_info(self):
        """
        恢复当前层的权重以及 mask
        :return:
        """
        # 改变weight，首先提取出参数，然后，初始化weight，再赋值
        if self.save_weight_data is not None:
            self.weight = paddle.create_parameter(shape=self.save_weight_data.shape, dtype='float32')
            self.weight.set_value(self.save_weight_data)
            if self.save_weight_grad is not None:
                self.weight.grad = self.save_weight_grad

        self.mask = self.save_mask

        if self.bias is not None:
            self.bias = paddle.create_parameter(shape=self.save_bias_data.shape, dtype='float32')
            self.weight.set_value(self.save_bias_data)
            if self.save_bias_grad is not None:
                self.bias.grad = self.save_bias_grad
            if self.save_bias_mask is not None:
                self.bias_mask = self.save_bias_mask

        self.out_features = self.weight.shape[0]
        self.in_features = self.weight.shape[1]

    def prune_out_channels(self, ind):
        self.save_layer_info()

        # 改变weight，首先提取出参数，然后，初始化weight，再赋值 (paddle 多于 ind 大于 weight 的情况，进行了补1)
        temp = paddle.index_select(self.weight.clone(), ind, 0)
        self.weight = paddle.create_parameter(shape=temp.shape, dtype='float32')
        self.weight.set_value(temp)

        # 有梯度的话需要记录梯度，tensor类型无需特别地创建
        if self.weight.grad is not None:
            self.weight.grad = paddle.index_select(self.weight.grad, ind, 0)

        # mask也需要进行裁剪
        self.mask = paddle.index_select(self.mask, ind, 0)

        # 乘一下置零，可有可无
        self.weight.set_value(self.weight * self.mask)

        # 出口还需要裁一下bias
        if self.bias is not None:
            temp = paddle.index_select(self.bias.clone(), ind, 0)
            self.bias = paddle.create_parameter(shape=temp.shape, dtype='float32')
            self.bias.set_value(temp)

            if self.bias_mask is not None:
                self.bias_mask = paddle.index_select(self.bias_mask, ind, 0)
            if self.bias.grad is not None:
                self.bias.grad = paddle.index_select(self.bias.grad, ind, 0)

        # 记录下出口通道
        self.out_features = len(ind)

    def prune_in_channels(self, ind):
        self.save_layer_info()

        # 改变weight，首先提取出参数，然后，初始化weight，再赋值, paddle 的 linear 入口在第一位
        temp = paddle.index_select(self.weight.clone(), ind, 0)
        self.weight = paddle.create_parameter(shape=temp.shape, dtype='float32')
        self.weight.set_value(temp)

        # 有梯度的话需要记录梯度，tensor类型无需特别地创建
        if self.weight.grad is not None:
            self.weight.grad = paddle.index_select(self.weight.grad, ind, 0)

        # mask也需要进行裁剪, 记录下入口通道数
        self.mask = paddle.index_select(self.mask, ind, 0)
        self.in_features = len(ind)

        # 乘一下置零，可有可无
        self.weight.set_value(self.weight * self.mask)

    @property
    def num_weight(self):
        return int(paddle.sum(self.mask).item() + paddle.sum(self.bias_mask).item())

if __name__=="__main__":
    a = DenseLinear(512, 512, use_mask=True)
    rank = paddle.linspace(1, 64, num=64)