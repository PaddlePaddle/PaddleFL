export GLOG_v=3
wget ${FL_MASTER_SERVICE_HOST}:${FL_MASTER_SERVICE_PORT_FL_MASTER}/fl_job_config.tar.gz
while [ $? -ne 0 ]
do
    sleep 3
    wget ${FL_MASTER_SERVICE_HOST}:${FL_MASTER_SERVICE_PORT_FL_MASTER}/fl_job_config.tar.gz
done
tar -xf fl_job_config.tar.gz
python -u fl_server.py > server.log 2>&1 
