#!/bin/bash
unset http_proxy
unset https_proxy
ps -ef | grep -E fl_ | grep -v grep | awk '{print $2}' | xargs kill -9

log_dir=${1:-"logs"}
mkdir -p ${log_dir}

python fl_master.py > ${log_dir}/master.log 2>&1 &
sleep 2
python -u fl_scheduler.py > ${log_dir}/scheduler.log 2>&1 &
sleep 5
python -u fl_server.py > ${log_dir}/server0.log 2>&1 &
sleep 2
for ((i=0;i<4;i++))
do
    python -u fl_trainer.py $i > ${log_dir}/trainer$i.log 2>&1 &
    sleep 2
done
