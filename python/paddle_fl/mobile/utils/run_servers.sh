endpoint=("50050" "50051" "50052" "50053" "50054" "50055" "50056" "50057" "50058" "50059")

for i in {0..9}
do
    python data_server_impl.py ${endpoint[$i]} &
done
