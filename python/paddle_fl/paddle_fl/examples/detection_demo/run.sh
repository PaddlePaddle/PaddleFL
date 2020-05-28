wget --no-check-certificate https://paddlefl.bj.bcebos.com/detection_programs/faster_rcnn_program.tar.gz
tar -xf faster_rcnn_program.tar.gz

unset http_proxy
unset https_proxy
export PYTHONPATH=/path/to/PaddleDetection
CUDA_VISIBLE_DEVICES=0,1

python fl_master.py
sleep 2
python -u fl_scheduler.py >scheduler.log &
sleep 2 
python -u fl_server.py >server0.log &
sleep 2
python -u fl_trainer.py 0 > 5_trainer0.log &
sleep 2
python -u fl_trainer.py 1 > 5_trainer1.log &
sleep 2
