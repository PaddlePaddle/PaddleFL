import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=500,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--global_momentum', type=int, default=0,
                        help="use global_momentum to update 1 for yes")

    # share arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument("--central_train", type=int, default=1, help="Default set to non central train. Set to 1 for central train")
    parser.add_argument("--share_percent", type=int, default=0, help="the shared data percent")
    parser.add_argument("--share_l", type=int, default=5, help="the noniid degree of shared data")
    parser.add_argument("--decay", type=float, default=0.99, help="decay for central training")

    # model arguments
    parser.add_argument('--model', type=str, default='lenet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")


    # prune arguments
    parser.add_argument("--prune_interval", type=int, default=-1, help="the interval adjacent two pruning")
    parser.add_argument("--prune_rate", type=float, default=0.6, help="the rate for pruning")
    parser.add_argument("--auto_rate", type=int, default=0, help="Default set to non auto rate. Set to 1 for auto rate")

    # prox agruments
    parser.add_argument("--server_mu", type=float, default=0.0, help="the server norm param")
    parser.add_argument("--client_mu", type=float, default=0.0, help="the client norm param")
    parser.add_argument("--auto_mu", type=int, default=0, help="Default set to non auto mu. Set to 1 for auto mu")

    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optim', type=str, default='sgd', help="type \
                        of optim")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument("--result_dir", type=str, help="dir name for save result", required=True)

    args = parser.parse_args()
    return args