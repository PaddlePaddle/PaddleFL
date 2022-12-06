export TEST_REDIS_IP="127.0.0.1"
export TEST_REDIS_PORT="6379"
nohup python3 mid3_C0.py >> ./logs/console0.log 2>>./logs/error0.log &
nohup python3 mid3_C1.py >> ./logs/console1.log 2>>./logs/error1.log &
nohup python3 mid3_C2.py >> ./logs/console2.log 2>>./logs/error2.log &
