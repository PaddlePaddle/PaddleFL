from paddle_fl.core.scheduler.agent_master import FLScheduler

worker_num = 5
server_num = 1
scheduler = FLScheduler(worker_num,server_num)
scheduler.set_sample_worker_num(5)
scheduler.init_env()
print("init env done.")
scheduler.start_fl_training()
