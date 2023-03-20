#!/bin/bash
declare -a port
# echo $1
# echo $2
for ((i=$1;i<=$2;i++))
do
        port[$i]=$i
        kill -9 ${port[$i]}
        echo " kill ${port[$i]} ..."
done