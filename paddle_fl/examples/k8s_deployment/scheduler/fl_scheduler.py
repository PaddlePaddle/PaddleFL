import argparse
from paddle_fl.core.scheduler.agent_master import FLScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="scheduler")
    parser.add_argument(
        '--trainer_num',
        type=int,
        default=2,
        help='number trainers(default: 2)')

    return parser.parse_args()


args = parse_args()
num_trainer = args.trainer_num
worker_num = num_trainer
server_num = 1
# Define the number of worker/server and the port for scheduler
scheduler = FLScheduler(worker_num, server_num, port=9091)
scheduler.set_sample_worker_num(worker_num)
scheduler.init_env()
print("init env done.")
scheduler.start_fl_training()
