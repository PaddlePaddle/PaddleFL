import paddle
import paddle.nn.functional as F

import numpy as np

class DenseConv2d(paddle.nn.Conv2D):
    """
    为卷积层包装上一层 mask 变量，让其方便进行剪枝
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True,
                 padding_mode='zeros', mask: paddle.Tensor = None, use_mask=False):
        super(DenseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                          dilation, groups, padding_mode, bias_attr=use_bias)
        if mask is None:
            self.mask = paddle.ones_like(self.weight)
        else:
            self.mask = mask
            assert self.mask.shape == self.weight.shape

        self.use_mask = use_mask

        self.ind = None
        self.bias_mask = paddle.ones_like(self.bias)

        self.rank_mask = None
        self.use_bias = use_bias

        self.save_weight_data = None
        self.save_weight_grad = None

        self.save_bias_data = None
        self.save_bias_grad = None

        self.save_mask = None
        self.save_bias_mask = None

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # self._initial_weight = self.weight.clone()
        # self._initial_bias = self.bias.clone() if isinstance(self.bias, paddle.Tensor) else None



    def forward(self, inp):
        """
        前向传播前，将对应 mask 位置的权重设置为 0
        :param inp:
        :return:
        """
        masked_weight = self.weight * self.mask if self.use_mask else self.weight
        masked_bias = self.bias

        if self.use_bias and self.bias is not None and self.bias_mask is not None:
            masked_bias = masked_bias * self.bias_mask

        return self.conv2d_forward(inp, masked_weight, masked_bias)

    def conv2d_forward(self, input, weight, bias):
        """
        进行前向传播
        :param input:
        :param weight:
        :param bias:
        :return:
        """
        # if self.padding_mode == 'circular':
        #     expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
        #                         (self.padding[0] + 1) // 2, self.padding[0] // 2)
        #     return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
        #                     weight, bias, self.stride,
        #                     _pair(0), self.dilation, self.groups)
        # else:
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def prune_by_threshold(self, thr):
        """
        小于某个阈值的 mask 全都设置为 0
        :param thr:
        :return:
        """
        self.mask *= (paddle.abs(self.weight) >= thr)

    def retain_by_threshold(self, thr):
        """
        这不是一样吗
        :param thr:
        :return:
        """
        self.mask *= (paddle.abs(self.weight) >= thr)

    def prune_by_rank(self, rank):
        """
        根据 rank 值来进行裁剪，将权重从小到大排序，第 rank 个的值作为裁剪的阈值
        :param rank:
        :return:
        """
        if rank == 0:
            return
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = paddle.sort(paddle.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def retain_by_rank(self, rank):
        """
        降序排列，第 rank 个的值作为保留的阈值
        :param rank:
        :return:
        """
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = paddle.sort(paddle.abs(weights_val), descending=True)[0]
        thr = sorted_abs_weights[rank]
        self.retain_by_threshold(thr)

    def prune_by_pct(self, pct):
        """
        根据一定的百分比，计算出 rank，之后裁剪
        :param pct:
        :return:
        """
        if pct == 0:
            return
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        """
        根据比例，随机裁剪
        :param pct:
        :return:
        """
        prune_idx = int(self.num_weight * pct)
        rand = paddle.rand_like(self.mask, device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = paddle.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    def move_data(self):
        # self.mask = self.mask
        # self.bias_mask = self.bias_mask
        paddle.device.set_device('gpu')

    def unstructured_by_rank(self, rank, rate, device: paddle.device, mode=1):
        """
        根据 rank 进行非结构化剪枝
        :param rank:
        :param rate:
        :param device:
        :return:
        """
        f, _, _, _ = self.weight.shape
        min_rank, max_rank = paddle.min(rank), paddle.max(rank)
        pruned_num = (max_rank - min_rank) * rate + min_rank
        # pruned_count = int(rate * f)
        pruned_count = (rank <= pruned_num).sum()
        if mode == 1:
            pruned_count = int(rate * f)

        print("该层剪枝率：{}".format(pruned_count / f))
        ind = paddle.argsort(rank)[pruned_count:]  # preserved filter id
        ind = paddle.sort(ind)

        self.ind = ind

        ones_re, zeros_re = paddle.ones_like(rank), paddle.zeros_like(rank)
        rank_mask = paddle.where(rank > pruned_num, ones_re, zeros_re)
        rank_mask = rank_mask > 0

        rank_mask = rank_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.rank_mask = rank_mask
        self.uprune_out_channels(rank_mask, device)

    def structured_by_rank(self, rank, rate, device: paddle.device, mode=1):
        """
        根据 rank 进行结构化剪枝
        :param rank:
        :param rate:
        :param device:
        :param mode: 0 表示以最大值和最小值的之间某个百分比进行阈值裁剪， 1 表示直接裁剪百分比数量的通道
        :return:
        """
        f, _, _, _ = self.weight.shape
        min_rank, max_rank = paddle.min(rank), paddle.max(rank)
        pruned_num = (max_rank - min_rank) * rate + min_rank
        pruned_count = (rank <= pruned_num).sum()
        if mode == 1:
            pruned_count = int(rate * f)

        print("该层剪枝率：{}".format(pruned_count / f))
        ind = paddle.argsort(rank)[pruned_count:] # preserved filter id
        ind = paddle.sort(ind)

        self.ind = ind

        self.prune_out_channels(ind)

    def uprune_out_channels(self, rank_mask, device: paddle.device):
        """
        实际的非结构化剪枝，更新 mask，以及对应位置的权重置 0
        :param rank_mask:
        :param device:
        :return:
        """
        print(self.mask.shape, rank_mask.shape)

        if rank_mask is not None:
            self.mask *= rank_mask

        if self.bias is not None and rank_mask is not None:
            if self.bias_mask is None:
                self.bias_mask = paddle.zeros_like(self.bias)
            self.bias_mask *= rank_mask.squeeze()

    def recovery_mask(self):
        """
        恢复 mask
        :return:
        """
        self.mask = paddle.ones_like(self.weight)
        self.bias_mask = paddle.ones_like(self.bias)


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

        self.out_channels = self.weight.shape[0]
        self.in_channels = self.weight.shape[1]

    def prune_out_channels(self, ind):
        """
        根据卷积层的索引，裁剪掉对应的出口 filter, 权重和 mask 都会进行实际裁剪
        :param ind:
        :return:
        """
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
        self.out_channels = len(ind)

    def prune_in_channels(self, ind):
        """
        根据卷积层的索引，裁剪掉对应的入口 filter，权重和 mask 都会进行实际裁剪
        :param ind:
        :return:
        """
        self.save_layer_info()

        # 改变weight，首先提取出参数，然后，初始化weight，再赋值
        temp = paddle.index_select(self.weight.clone(), ind, 1)
        self.weight = paddle.create_parameter(shape=temp.shape, dtype='float32')
        self.weight.set_value(temp)

        # 有梯度的话需要记录梯度，tensor类型无需特别地创建
        if self.weight.grad is not None:
            self.weight.grad = paddle.index_select(self.weight.grad, ind, 1)

        # mask也需要进行裁剪, 记录下入口通道数
        self.mask = paddle.index_select(self.mask, ind, 1)
        self.in_channels = len(ind)

        # 乘一下置零，可有可无
        self.weight.set_value(self.weight * self.mask)

    @property
    def num_weight(self):
        return int(paddle.sum(self.mask).item() + paddle.sum(self.bias_mask).item())

if __name__=="__main__":
    f = [[[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]],
         [[1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]],
         [[1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]]
         ]
    paddle.device.set_device('gpu')  # or 'gpu'
    a = DenseConv2d(3, 64, kernel_size=3, padding=1, use_mask=True)
    rank = paddle.linspace(1, 64, num=64)
    a.unstructured_by_rank(rank, 0.6, "cpu")
    a.recovery_mask()
    ind = a.ind
    print(a.num_weight)
    a.prune_out_channels(ind)
    print(a.ind.shape[0])
    a.structured_by_rank(rank, 0.6, "cpu")
    e = np.array(f)
    g = paddle.index_select(paddle.to_tensor(e), paddle.to_tensor([1, 2]), 1)
    print(g)
    print(g.shape)
    mask = paddle.ones_like(g)
    print(mask)
    mask *= (g > 3)
    print(mask)
    g_val = g[mask == 1]
    print(g_val)
    g_val_sort = paddle.sort(g_val)
    print(g_val_sort)
    print(g_val_sort[0])
    print(g_val.unsqueeze(-1).unsqueeze(-1).squeeze())