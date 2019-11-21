python fl_master.py
sleep 2
python -u fl_scheduler.py >scheduler.log &
sleep 2
python -u fl_server.py >server0.log &
sleep 2
python -u fl_trainer.py 0 >trainer0.log &
sleep 2
python -u fl_trainer.py 1 >trainer1.log &
sleep 2
python -u fl_trainer.py 2 >trainer2.log &
sleep 2
python -u fl_trainer.py 3 >trainer3.log &
