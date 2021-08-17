# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


class LayerBase(paddle.nn.Layer):

    def __init__(self):
        super(LayerBase, self).__init__()

    @paddle.jit.to_static
    def forward(self, **feed):
        raise NotImplementedError("Failed to run forward")

    def get_fetch_vars(self):
        raise NotImplementedError("Failed to get fetch vars")

    def get_loss(self, inputs, predict):
        raise NotImplementedError("Failed to get loss")
