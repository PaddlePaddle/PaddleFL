import multiprocessing
import leveldb
import sys
import os
from agent_master import *

def task_func(task_info):
    def init_scheduler():
        worker_num = 10
        server_num = 10
        scheduler = FLScheduler(worker_num, server_num)
        scheduler.set_sample_worker_num()
        scheduler.init_env()
        print("init env done.")
        scheduler.start_fl_training()

    def init_worker():
        agent = FLWorkerAgent("127.0.0.1:9091", "127.0.0.1:{}".format(9000 + task_info[0]))
        agent.connect_scheduler()
        print("connected")
        import time
        time.sleep(3)

        for i in range(10):
            if agent.can_join_training():
                # do some training here
                time.sleep(3)
                agent.finish_training()
            else:
                print("rejected")
                time.sleep(3)
            print("round {} finished".format(i))

    def init_server():
        agent = FLServerAgent("127.0.0.1:9091", "127.0.0.1:{}".format(9000 + task_info[0]))
        agent.connect_scheduler()

    if task_info[1] == 0:
        init_scheduler()
    elif task_info[1] == 1:
        init_worker()
    else:
        init_server()

pool = multiprocessing.Pool(processes=21)
port_index = 1
task_info = []
task_info.append([port_index, 0])
port_index += 1
for i in range(10):
    task_info.append([port_index, 1])
    port_index += 1
for i in range(10):
    task_info.append([port_index, 2])
    port_index += 1

results = pool.map(task_func, task_info)
pool.close()
pool.join()
