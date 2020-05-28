# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy as np
import sys
import paddle
import pickle
import paddle.fluid as fluid
import logging
import math
import unittest
import os
from ppdet.data.source.voc import VOCDataSet
from ppdet.data.reader import Reader
from ppdet.utils.download import get_path
from ppdet.utils.download import DATASET_HOME

from ppdet.data.transform.operators import DecodeImage, RandomFlipImage, NormalizeImage, ResizeImage, Permute
from ppdet.data.transform.batch_operators import PadBatch

trainer_id = int(sys.argv[1])  # trainer id for each guest


class DataReader():
    def __init__(self):
        """ setup
        """
        self.root_path = '/path/to/your/fl_fruit'
        self.anno_path = '/path/to/your/fl_fruit/train' + str(
            trainer_id) + '.txt'
        self.image_dir = '/path/to/your/fl_fruit/JPEGImages'

    def tearDownClass(self):
        """ tearDownClass """
        pass

    def test_loader(self):
        coco_loader = VOCDataSet(
            dataset_dir=self.image_dir,
            image_dir=self.root_path,
            anno_path=self.anno_path,
            sample_num=240,
            use_default_label=False,
            label_list='/path/to/your/fl_fruit/label_list.txt')
        sample_trans = [
            DecodeImage(to_rgb=True), RandomFlipImage(), NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                is_scale=True,
                is_channel_first=False), ResizeImage(
                    target_size=800, max_size=1333, interp=1),
            Permute(to_bgr=False)
        ]
        batch_trans = [PadBatch(pad_to_stride=32, use_padded_im_info=True), ]

        inputs_def = {
            'fields':
            ['image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd'],
        }
        data_loader = Reader(
            coco_loader,
            sample_transforms=sample_trans,
            batch_transforms=batch_trans,
            batch_size=1,
            shuffle=True,
            drop_empty=True,
            inputs_def=inputs_def)()

        return data_loader


job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform scheduler IP address to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
trainer.start(fluid.CUDAPlace(trainer_id))

test_program = trainer._main_program.clone(for_test=True)

image = fluid.layers.data(
    name='image', shape=[3, None, None], dtype='float32', lod_level=0)
im_info = fluid.layers.data(
    name='im_info', shape=[None, 3], dtype='float32', lod_level=0)
im_id = fluid.layers.data(
    name='im_id', shape=[None, 1], dtype='int64', lod_level=0)
gt_bbox = fluid.layers.data(
    name='gt_bbox', shape=[None, 4], dtype='float32', lod_level=1)
gt_class = fluid.layers.data(
    name='gt_class', shape=[None, 1], dtype='int32', lod_level=1)
is_crowd = fluid.layers.data(
    name='is_crowd', shape=[None, 1], dtype='int32', lod_level=1)
place = fluid.CUDAPlace(trainer_id)
feeder = fluid.DataFeeder(
    feed_list=[image, im_info, im_id, gt_bbox, gt_class, is_crowd],
    place=place)

output_folder = "5_model_node%d" % trainer_id
epoch_id = 0
step = 0

para_dir = "faster_rcnn_program"

while not trainer.stop():
    epoch_id += 1
    if epoch_id > 120:
        break
    print("epoch %d start train" % (epoch_id))
    test_class = DataReader()
    data_loader = test_class.test_loader()
    for step_id, data in enumerate(data_loader):
        acc = trainer.run(feeder.feed(data), fetch=['sum_0.tmp_0'])
        step += 1
        print("step: {}, loss: {}".format(step, acc))

    if trainer_id == 0:
        save_dir = (output_folder + "/epoch_%d") % epoch_id
        trainer.save(para_dir, save_dir)
