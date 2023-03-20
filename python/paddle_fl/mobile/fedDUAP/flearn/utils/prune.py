import paddle
from paddle.io import DataLoader
from flearn.utils.update import DatasetSplit
from pyhessian import hessian
import numpy as np
from data.cifar10.cifar10_data import get_dataset
from flearn.utils.model_utils import is_conv
from flearn.models import cnn, vgg, resnet, lenet

def model_eva(model, train_loader):
    """
    执行前向传播，用于获取 hrank 方法中的特征图
    :param model:
    :param train_loader:
    :return:
    """
    server_inputs, server_outputs = [], []

    for batch_idx, (inp, out) in enumerate(train_loader):
        server_inputs.append(inp)
        server_outputs.append(out)
        if batch_idx >= 4:
            break

    for _ in range(1, 1 + 1):
        model.evaluate(zip(server_inputs, server_outputs))


def get_rank(model, feature_result_list, total_list, train_loader, layer_idx=0):
    """
    根据模型自身参数，以及一定量的数据，得到特征图，并根据特征图，得到该层每个通道的平均特征图秩
    :param model:
    :param feature_result_list:
    :param total_list:
    :param train_loader:
    :param rankIdx:
    :return:
    """
    def get_feature_hook(self, input, output):
        nonlocal feature_result
        nonlocal total
        a = output.shape[0]  # 输入样本量
        b = output.shape[1]  # 输出通道个数
        if len(output.shape) > 2:
            # 对每个样本，每个通道求特征图的秩，并进行堆叠。之后变化为 （a, b) 的形式，通过纵向求和，就可以得到每个通道的累加秩
            c = paddle.to_tensor([paddle.linalg.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])
            c = paddle.reshape(c, [a, -1]).astype(paddle.float32)
            # c = c.view(a, -1).float()
            c = c.sum(0)

        # 多个 batch 进行加权平均
        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total

    layer, layer_perfix = model.relu_layers[layer_idx], model.relu_layers_prefixes[layer_idx]

    feature_result = feature_result_list[layer_idx]
    total = total_list[layer_idx]

    handler = layer.register_forward_post_hook(get_feature_hook)
    # handler = layer.register_forward_hook(get_feature_hook)
    model_eva(model, train_loader)

    feature_result_list[layer_idx] = feature_result
    total_list[layer_idx] = total
    handler.remove()

    return feature_result_list

def hrank_prune(model, dataset, idxs, prune_rate, layer_idx=0, device="cpu"):
    """
    使用 hrank 的特征图方式进行结构化剪枝
    :param model:
    :param dataset:
    :param idxs:
    :param prune_rate:
    :param layer_idx:
    :param device:
    :return:
    """
    print(f"=== Pruning === Current pruned layer {layer_idx} using prune rate {prune_rate}")
    train_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=100, shuffle=True)
    feature_result_list = [paddle.to_tensor(0.)] * len(model.relu_layers)
    total_list = [paddle.to_tensor(0.)] * len(model.relu_layers)
    # 将特征图秩保存在 feature_result_list 中
    get_rank(model, feature_result_list, total_list, train_loader, layer_idx)
    _, ind = model.unstructured_by_rank(feature_result_list, prune_rate, layer_idx, device)
    return ind

def make_new_mask(model, prune_ratio, device):
    """
    根据剪枝率，得到模型剪枝的 mask
    :param model:
    :param prune_ratio:
    :param device:
    :return:
    """
    weights = {k: v.clone().numpy()
               for k, v in model.named_parameters()
               if not v.stop_gradient}

    mask = {k: np.ones_like(v)
            for k, v in weights.items()}

    # for k, v in model.named_parameters():
    #     print(k)
    # # print(len(weights.keys()))
    # for k,v in net.named_modules():
    #     print(k)

    # flat the weights
    weight_flat = np.concatenate([v.flatten() for k, v in weights.items()])

    # get the thredsheld
    number_of_weights_to_prune = int(np.ceil(prune_ratio * weight_flat.shape[0]))
    threshold = np.sort(np.abs(weight_flat))[number_of_weights_to_prune]

    # get the prune mask
    new_mask = {k: np.where(np.abs(v) > threshold, mask[k], np.zeros_like(v))
                for k, v in weights.items()}
    inv_mask = {k: np.where(np.abs(v) > threshold, np.zeros_like(v), mask[k])
                for k, v in weights.items()}

    # check sparsity of new_mask
    n_elements = np.sum([np.sum(v) for v in new_mask.values()])
    print("sparsity of new_mask: {}".format(n_elements / len(weight_flat)))
    new_mask = {k: paddle.to_tensor(v).astype(paddle.int32)
                for k, v in new_mask.items()}
    inv_mask = {k: paddle.to_tensor(v).astype(paddle.int32)
                for k, v in inv_mask.items()}

    return new_mask, inv_mask

def get_rate_for_each_layers(model, global_prune_rate):
    """
    根据模型权重的范式以及全局剪枝率，为每一层卷积层确定剪枝率
    :param model:
    :param global_prune_rate:
    :return:
    """
    weights = {k: v.clone().numpy()
               for k, v in model.named_parameters()
               if not v.stop_gradient}
    # flat the weights
    weight_flat = np.concatenate([v.flatten() for k, v in weights.items()])
    # get the thredsheld
    number_of_weights_to_prune = int(np.ceil(global_prune_rate * weight_flat.shape[0]))
    threshold = np.sort(np.abs(weight_flat))[number_of_weights_to_prune]
    compress_rate = []
    for i, layer in enumerate(model.prunable_layers):
        if is_conv(layer):
            a = np.abs(layer.weight.clone().numpy()) < threshold
            b = int(a.sum())
            c = layer.weight.size
            compress_rate.append(round(b / c, 4))

    return compress_rate

def lt_mask_copy(model, mask):
    """
    将 lt 的得到的 mask 覆盖到 model 上
    :param model:
    :param mask:
    :return:
    """
    for k, v in mask.items():
        if "bn" in k or "downsample" in k:
            continue
        split = k.split('.')
        split.pop()
        prefix = ".".join(split)
        layer_prefix_idx = model.prunable_layer_prefixes.index(prefix)
        layer = model.prunable_layers[layer_prefix_idx]
        if k.endswith("weight"):
            layer.mask = v.clone() * layer.mask
            layer.mask.stop_gradient = True

        if k.endswith("bias"):
            layer.bias_mask = v.clone() * layer.bias_mask
            layer.bias_mask.stop_gradient = True

def lt_prune(model, pct=0.6, device="cpu"):
    """
    根据剪枝率从绝对值小权重开始进行剪枝，非结构化剪枝，只是改变 mask
    :param model:
    :param pct:
    :param device:
    :return:
    """
    new_mask, inv_mask = make_new_mask(model, pct, device)
    lt_mask_copy(model, new_mask)
    return model

def _gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

def _density_generate(eigenvalues, weights, num_bins=10000, sigma_squared=1e-5, overhead=0.01):
    """
    生成特征密度网格
    :param eigenvalues:
    :param weights:
    :param num_bins:
    :param sigma_squared:
    :param overhead:
    :return:
    """

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = _gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids

def _get_ratio(t, density_eigen, density_weight):
    """
    根据特征值获取剪枝率
    :param t:
    :param density_eigen:
    :param density_weight:
    :return:
    """
    density, grids = _density_generate(density_eigen, density_weight)
    sum = 0
    for i in range(len(grids - 1) - 1):
        if grids[i + 1] <= t:
            sum += density[i] * (grids[i + 1] - grids[i])
            i += 1
    ratio = 1 - sum
    # print("t: {:.6f}   - prune ratio: {:.4f}".format(t, ratio))
    return ratio

def get_ratio(model, dataset, idxs):
    """
    根据特征值的概率密度，获取剪枝率
    :param model:
    :param dataset:
    :param idxs:
    :param device:
    :return:
    """
    print("=== 获取剪枝率 ===")
    model.eval()
    from flearn.utils.pt_update import DatasetSplit as pt_DatasetSplit
    from paddle.io import DataLoader as pt_DataLoader
    import paddle
    device = "gpu" if paddle.device.get_device() else "cpu"
    train_loader = pt_DataLoader(pt_DatasetSplit(dataset, idxs), batch_size=100, shuffle=True)
    for data in train_loader:
        inputs, targets = data
        break

    hessian_comp = hessian(model, model.loss_func, data=(inputs, targets), cuda=True if device != "cpu" else False)
    density_eigen, density_weight = hessian_comp.density(iter=50, n_v=1)
    inc = 0.1
    while True:
        t = 0.000
        ratios = []
        flag = False
        while True:
            ratio = _get_ratio(t, density_eigen=density_eigen, density_weight=density_weight)
            ratios.append(ratio)
            if ratio < ratios[0] / 2:
                break
            if len(ratios) >= 4:
                if abs(ratios[-1] - ratios[-2]) < 0.005 and abs(ratios[-2] - ratios[-3]) < 0.005 and abs(ratios[-3] - ratios[-4]) < 0.005:
                    flag = True
                    break
            t += inc
        if flag:
            break
        inc /= 2
    return ratios[-1]

if __name__=="__main__":
    import random
    paddle.seed(777)
    np.random.seed(777)
    random.seed(777)

    # output = paddle.ones([2, 3])
    # r = paddle.linalg.matrix_rank(output).item()

    # train_dataset, test_dataset, user_groups = get_dataset(num_data=40000, num_users=100, iid=False, num_share=4000,
    #                                                        l=2, unequal=True)
    model = resnet.resnet18()

    # from data.cifar10.cifar10_data_pt import get_pt_dataset
    # pt_dataset = get_pt_dataset()

    # ratio = get_ratio(model.get_pt_model(), pt_dataset, user_groups[100])
    idxs = np.array(range(100))
    # print(f"ratio: {ratio}")

    # get_rate_for_each_layers(model, 0.6)

    # hrank_prune(model, train_dataset, idxs, 0.6, layer_idx=2, device="cpu")
    # sd = model.state_dict()
    # lt_prune(model)
    lt_prune(model)
    print(model.calc_num_all_active_params())

    # lt_prune(vgg.VGG11())
    # lt_prune(resnet.resnet18())
    # lt_prune(lenet.LENET())