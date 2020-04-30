wget ${FL_MASTER_SERVICE_HOST}:${FL_MASTER_SERVICE_PORT_FL_MASTER}/fl_job_config.tar.gz
while [ $? -ne 0 ]
do
    sleep 3
    wget ${FL_MASTER_SERVICE_HOST}:${FL_MASTER_SERVICE_PORT_FL_MASTER}/fl_job_config.tar.gz
done
tar -xf fl_job_config.tar.gz
sleep 10
python -u fl_trainer.py 0 
