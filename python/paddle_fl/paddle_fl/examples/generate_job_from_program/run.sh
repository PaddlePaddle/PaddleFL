unset http_proxy
unset https_proxy
python program_saver.py

python fl_master.py
sleep 2
python -u fl_scheduler.py >scheduler.log &
sleep 2
python -u fl_server.py >server0.log &
sleep 2
python -u fl_trainer.py 0 >trainer0.log &
sleep 2
python -u fl_trainer.py 1 > trainer1.log &
sleep 2
