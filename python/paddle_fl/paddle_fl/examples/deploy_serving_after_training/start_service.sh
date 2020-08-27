model_dir=$1
python -m paddle_serving_server.serve --model $model_dir --thread 10 --port 9292 &
