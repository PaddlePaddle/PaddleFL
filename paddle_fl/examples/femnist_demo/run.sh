#killall python
#python fl_master.py
#sleep 2
python -u fl_server.py >log/server0.log &
sleep 2
python -u fl_scheduler.py >scheduler.log &
sleep 2
python -u fl_server.py >server0.log &
sleep 2
for ((i=0;i<4;i++))
do
python -u fl_trainer.py $i >trainer$i.log &
sleep 2
done
