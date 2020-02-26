#Download config file 
wget

#Download image
sudo docker pull [paddle-fl image]

#Build docker 
sudo docker run --name paddlefl -it -v $PWD:/root [paddle-fl image] /bin/bash

sudo docker cp /path/to/config paddlefl:/path/to/config/file/at/container

#Run program

python -u fl_trainer.py > trainer.log &
