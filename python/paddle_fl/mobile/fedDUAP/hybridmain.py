"""
Hybrid main
"""
from flearn.experiments.hybrid import *
from flearn.utils.options import args_parser

def main():
    """
    main
    :return:
    """
    args = args_parser()
    share_percent = args.share_percent
    prune_interval = args.prune_interval
    result_dir = args.result_dir
    iid = True if args.iid == 1 else False
    unequal = True if args.unequal == 1 else False
    prune_rate = args.prune_rate
    auto_rate = True if args.auto_rate == 1 else False
    auto_mu = True if args.auto_mu == 1 else False
    server_mu = args.server_mu
    client_mu = args.client_mu

    t = Hybrid(args, share_percent=share_percent, iid=iid, unequal=unequal, result_dir=result_dir)
    t.train()

if __name__ == '__main__':
    main()