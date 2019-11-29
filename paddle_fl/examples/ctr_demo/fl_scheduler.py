from paddle_fl.core.scheduler.agent_master import FLScheduler

worker_num = 2
server_num = 1
scheduler = FLScheduler(worker_num,server_num)
scheduler.set_sample_worker_num(worker_num)
scheduler.init_env()
print("init env done.")
scheduler.start_fl_training()
