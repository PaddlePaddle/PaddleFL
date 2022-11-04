"""
Simulate the job running process
to obtain the time required and the devices selected in per round for each job
"""

import copy
import json
import logging
import multiprocessing
import os
import time
import numpy as np
from core.scheduler import Common, Greedy, Bayesian, Fedcs, DRL, Genetic
from core.RL_scheduler import RL


def start(share_clients, share_count, lock, config, job, file, Pre_clients):
    if config["isIID"]:
        SD = "IID"
    else:
        SD = "NIID"
    #################################################################
    with open(file, "r") as f:
        client_time = json.load(f)
    Scheduler = config["scheduler"]

    filehandler = logging.FileHandler("results/" + Scheduler + "/" + file[-18:-5] + "/" + SD + "/" + job + '_time.log')
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    rounds_client = []
    simu_t = 0

    if config["scheduler"] == "DRL":
        model = RL(input_size=103, output_shape=100, episode=350, total_job=3)

    for r in range(config[job]["rounds"]):
        lock.acquire()
        if Scheduler == "Bayesian":
            clients = globals()[Scheduler](share_clients, share_count, config["client_per_round"],
                                           job_No=config[job]["No"],
                                           client_time=client_time, Pre_clients=Pre_clients)

        elif Scheduler == "DRL":
            print(config[job]["No"], np.sum(Pre_clients, axis=1))
            clients = globals()[Scheduler](model, share_clients, share_count, config["client_per_round"],
                                           job_No=config[job]["No"], Pre_clients=Pre_clients)

        else:
            clients = globals()[Scheduler](share_clients, share_count, config["client_per_round"],
                                           job_No=config[job]["No"],
                                           client_time=client_time, Pre_clients=Pre_clients)

        temp = Pre_clients[config[job]["No"]][:]
        for i in clients:
            temp[i] += 1
        Pre_clients.__setitem__(config[job]["No"], temp)

        share_clients[:] = list(set(share_clients).difference(set(clients)))
        lock.release()

        # Calculate the time of current round
        rounds_client.append(clients)
        simu_delta = max([client_time[i][config[job]["No"]] for i in clients])
        simu_t += simu_delta

        print(clients, simu_delta)
        # local train
        time.sleep(simu_delta/10)

        lock.acquire()
        share_clients[:] = list(set(share_clients).union(set(clients)))
        lock.release()

        logger.info('%s:: Epoch %d: simu_t=%.2f, simu_delta=%.2f' % (job, r, simu_t, simu_delta))

    with open("simulation/" + Scheduler + "/" + SD + "/" + job + "_rounds_client" + file[-6:],
              "w") as f:
        json.dump(rounds_client, f)
    lock.acquire()
    share_count.remove(config[job]["No"])
    lock.release()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    with open('config.json') as file:
        config = json.load(file)

    file = "utils/client_time_" + config["group"] + ".json"

    share_clients = multiprocessing.Manager().list([i for i in range(0, 100)])
    share_count = multiprocessing.Manager().list([0, 1, 2])
    lock = multiprocessing.Manager().Lock()
    Pre_clients = multiprocessing.Manager().list([[0 for i in range(100)] for j in range(config["total_jobs"])])

    Job = {"A": ["CnnEm", "VGG", "Lenet"], "B": ["CnnFM", "Resnet", "Alexnet"]}
    # start(share_clients, share_count, lock, config, "CnnEm", file, Pre_clients)

    p = multiprocessing.Pool(3)
    p.apply_async(start, args=(share_clients, share_count, lock, config, Job[config["group"]][0], file, Pre_clients))
    p.apply_async(start, args=(share_clients, share_count, lock, config, Job[config["group"]][1], file, Pre_clients))
    p.apply_async(start, args=(share_clients, share_count, lock, config, Job[config["group"]][2], file, Pre_clients))
    p.close()
    p.join()
