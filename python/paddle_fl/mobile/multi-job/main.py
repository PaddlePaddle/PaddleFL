"""
Read the configuration of each job
Create a process for each job
"""
import sys
import json
import os
import paddle
from core.Job import Image_CnnEm, Image_VGG, Image_LeNet, Image_CnnFM, Image_ResNet, Image_AlexNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
paddle.set_device('gpu')

job_n = int(sys.argv[1])

def main(Job_num):
    with open('config.json') as file:
        config = json.load(file)

    Job = {"A": ["Image_CnnEm", "Image_VGG", "Image_LeNet"], "B": ["Image_CnnFM", "Image_ResNet", "Image_AlexNet"]}

    globals()[Job[config["group"]][Job_num]](config)

if __name__ == '__main__':
    main(job_n)

