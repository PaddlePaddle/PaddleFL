# Example of a detection model training with FedAvg

This document introduce how to start a detection model training in PaddleFL with our pre-defined program. Now we only provide faster_rcnn, more models will be updated. 

### Dependencies

- paddlepaddle>=1.8
- paddle_fl>=1.0.1

Please use pip which has paddlepaddle installed

```sh
pip install paddle_fl
``` 


### How to Run

#### Download the dataset

```sh
# download and unzip the dataset
sh download.sh
```

#### Start training 

Before training, please modify the following paths according to your environment.

```python

# In run.sh, change the path to you PaddleDetection
export PYTHONPATH=/path/to/PaddleDetection

# In fl_train.py, change the path to your fl_fruit dataset that is downloaded in download.sh. 
# Note, the path should be absolute path rather than relative path. Otherwise, error will be raised. 

#line 41
self.root_path = '/path/to/your/fl_fruit'
#line 42
self.anno_path = '/path/to/your/fl_fruit/train' + str(trainer_id) + '.txt'
# line 44
self.image_dir = '/path/to/your/fl_fruit/JPEGImages'
# line 57
label_list='/path/to/your/fl_fruit/label_list.txt'

```

After modifying the path, you can run the following shell directly.

```sh

sh run.sh
```
