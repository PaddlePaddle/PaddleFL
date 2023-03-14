# FedDUAP Paddle

This is an Paddle implementation of the following paper: [FedDUAP: Federated Learning with Dynamic Update and Adaptive Pruning Using Shared Data on the Server](https://arxiv.org/pdf/2204.11536.pdf)

## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The model architecture. Options: `cnn`, `lenet`, `vgg`, `resnet` .|
| `dataset`      | Dataset to use. Options: `cifar10`. `cifar100`, `mnist`, `fashionmnist`|
| `local_bs` | Batch size. |
| `epochs` | Number of rounds of training. |
| `local_ep` | Number of local epochs. |
| `num_users` | Number of parties. |
| `frac` | The fraction of parties to be sampled in each round. |
| `comm_round`    | Number of communication rounds. |
| `iid` | Default set to IID. Set to `0` for non-IID. |
| `unequal` | Whether to use unequal data splits for non-i.i.d setting (use `0` for equal splits. |
| `prune_rate` | The rate for pruning. |
| `prune_interval` | The interval adjacent two pruning. Do not prune by setting `-1`.|
| `central_train` | Default set to non central train. Set to `1` for central train. |
| `share_percent` | The shared data percent. |
| `share_l` | The noniid degree of shared data. |
| `gpu` | To use cuda, set to a specific GPU ID. Default set to use CPU.. |
| `seed` | The initial seed. |

## Launch Experiments

Here is an example to run FedDUAP on CIFAR-10 with a simple CNN over IID data:

```
python ../centralmain.py --iid 0 --share_percent 10 --unequal 0 --dataset "cifar10" --central_train 1\
      --result_dir "FedDUAP_equal" --epochs 500 --local_bs 10 --local_ep 5 --decay 0.999\
      --prune_interval 30 --prune_rate 0.6 --share_l 5 --model "cnn" \
      
```

Here is an example to run FedDU on CIFAR-10 with a simple CNN over IID data:

```
python ../centralmain.py --iid 0 --share_percent 10 --unequal 0 --dataset "cifar10" --central_train 1\
      --result_dir "FedDU_equal" --epochs 500 --local_bs 10 --local_ep 5 --decay 0.999\
      --prune_interval -1 --prune_rate 0.6 --share_l 5 --model "cnn" \
      
```

Here is an example to run FedAP on CIFAR-10 with a simple CNN over IID data:

```
python ../centralmain.py --iid 0 --share_percent 10 --unequal 0 --dataset "cifar10" --central_train 0\
      --result_dir "FedAP_equal" --epochs 500 --local_bs 10 --local_ep 5 --decay 0.999\
      --prune_interval 30 --prune_rate 0.6 --share_l 5 --model "cnn" \
      
```