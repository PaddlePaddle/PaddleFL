unset http_proxy
unset https_proxy
python3 fl_master.py
sleep 2
python3 -u fl_server.py >server0.log &
sleep 2
python3 -u fl_trainer.py 0 >trainer0.log &
sleep 2
python3 -u fl_trainer.py 1 >trainer1.log &
