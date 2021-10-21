## Use the container
```
# Pull the pre-built image
docker pull quay.io/thinkall/paddlefl:1.1.2  

# Build the image by yourself
docker build -t paddlefl:1.1.2 .

# Launch
cd <project-root>
docker run --name <docker_name> --net=host -it -v $PWD:/paddle <image id> bash
```

## Run examples of paddle_fl
```
cd PaddleFL
docker run --name paddlefl --net=host -it -v $PWD:/paddle quay.io/thinkall/paddlefl:1.1.2 bash 

# Now in the terminal of container
root@docker-desktop:/paddle# cd python/paddle_fl/paddle_fl/examples/gru4rec_demo/
root@docker-desktop:/paddle/python/paddle_fl/paddle_fl/examples/gru4rec_demo# bash run.sh 
root@docker-desktop:/paddle/python/paddle_fl/paddle_fl/examples/gru4rec_demo# ps -ef | grep fl_
root        66     1  7 03:21 pts/0    00:00:01 python -u fl_scheduler.py
root       105     1 11 03:21 pts/0    00:00:01 python -u fl_server.py
root       171     1 31 03:21 pts/0    00:00:04 python -u fl_trainer.py 0
root       210     1 36 03:21 pts/0    00:00:04 python -u fl_trainer.py 1
root       249     1 42 03:21 pts/0    00:00:03 python -u fl_trainer.py 2
root       288     1 58 03:21 pts/0    00:00:04 python -u fl_trainer.py 3
root       747     1  0 03:21 pts/0    00:00:00 grep fl_
root@docker-desktop:/paddle/python/paddle_fl/paddle_fl/examples/gru4rec_demo# ls
README.md  download.sh  fl_job_config  fl_master.py  fl_scheduler.py  fl_server.py  fl_trainer.py  logs  mid_data  mid_data.tar  model_node4  run.sh  test.log
root@docker-desktop:/paddle/python/paddle_fl/paddle_fl/examples/gru4rec_demo# ls model_node4/
epoch_1  epoch_2  epoch_3  epoch_4  epoch_5
```

## Install new dependencies
Add the following to Dockerfile to make persist changes.
```
RUN apt update && apt install vim -y && apt clean
```
Or run them in the container to make temporary changes.