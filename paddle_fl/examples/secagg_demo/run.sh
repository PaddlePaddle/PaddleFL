unset http_proxy
unset https_proxy

if [ ! -d log ];then
    mkdir log
fi

python3 fl_master.py
sleep 2
python3 -u fl_server.py >log/server0.log &
sleep 2
python3 -u fl_trainer.py 0 >log/trainer0.log &
sleep 2
python3 -u fl_trainer.py 1 >log/trainer1.log &
