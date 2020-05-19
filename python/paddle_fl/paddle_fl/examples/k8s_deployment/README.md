# PaddleFL deployment example with Kubernetes

## How to run in PaddleFL

```sh
kubectl apply -f master.yaml
```

## Compile time

#### Master

```sh

#Define distributed training config for trainer and server
python fl_master.py --trainer_num 2
tar -zcvf fl_job_config.tar.gz fl_job_config

#Start HTTP server and wait download request from trainer and server
python -m SimpleHTTPServer 8000

```

## Run time

#### Scheduler
```sh

#Start a Scheduler
python fl_scheduler.py --trainer_num 2

```

#### Server
```sh

#Download job config file from master
wget ${FL_MASTER_SERVICE_HOST}:${FL_MASTER_SERVICE_PORT_FL_MASTER}/fl_job_config.tar.gz
tar -xf fl_job_config.tar.gz

#Start a Server
python -u fl_server.py > server.log 2>&1

```

#### Trainer
```sh

#Download job config file from master
wget ${FL_MASTER_SERVICE_HOST}:${FL_MASTER_SERVICE_PORT_FL_MASTER}/fl_job_config.tar.gz
tar -xf fl_job_config.tar.gz

#Start the ith trainer
python -u fl_trainer.py i

``` 
