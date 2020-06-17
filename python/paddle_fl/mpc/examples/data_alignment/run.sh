#!/bin/bash
ENDPOINTS=0:127.0.0.1:11111,1:127.0.0.1:22222

python align.py --party_id=0 --endpoints=$ENDPOINTS --data_file=data_0.txt 2>&1 >/dev/null &

python align.py --party_id=1 --endpoints=$ENDPOINTS --data_file=data_1.txt --is_receiver
