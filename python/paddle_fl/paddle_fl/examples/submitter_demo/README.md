# Example of submitting job to mpi cluster

This document introduces how to submit an FL job to mpi cluster

### Dependency

- paddlepaddle>=1.8
- paddle_fl==0.2.0

### How to install PaddleFL

Please use pip which has paddlepaddle installed

```sh
pip install paddle_fl==0.2.0
```

### How it works 

#### Prepare packages

- An executable python package that will be used in cluster
- An installl package of PaddldPaddle

#### Submitter job

The information of the cluster is defined in config.txt and will be transmitted into client.py. Then a function called job_generator() will generate job for fl_server and fl_trainer. Finally, the job will be submitted. 

The train_program.py is the executed program in cluster.
```sh
#use the python prepared above to generate fl job and submit the job to mpi cluster
python/bin/python client.py config.txt
```


