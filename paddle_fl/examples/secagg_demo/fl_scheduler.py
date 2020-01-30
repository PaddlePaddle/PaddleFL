from paddle_fl.core.scheduler.agent_master import FLScheduler

worker_num = 2
server_num = 1
# Define the number of worker/server and the port for scheduler
scheduler = FLScheduler(worker_num,server_num,port=9091)
scheduler.set_sample_worker_num(worker_num)
scheduler.init_env()
print("init env done.")
scheduler.start_fl_training()
