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
import logging
import paddle
from .layer_base import LayerBase

_LOGGER = logging.getLogger(__name__)


class HostLayerHandler(object):

    def __init__(self, layer, optimizer):
        if not isinstance(layer, LayerBase):
            raise TypeError(
                    "Failed: layer({}) must be subclass of LayerBase."
                    .format(type(layer)))
        self.layer = layer
        self.optimizer = optimizer
        
        self.fetch_vars = None # not thread safe, for backward

    def call_for_forward(self, **inputs):
        self.fetch_vars = self.layer(**inputs)
        if not isinstance(self.fetch_vars, (list, tuple)):
            self.fetch_vars = [self.fetch_vars]
        return self.fetch_vars

    def call_for_backward(self, grads, tensor_names_to_customer):
        temp_var = 0
        for idx, name in enumerate(tensor_names_to_customer):
            var = self.fetch_vars[idx]
            var_grad = grads["{}@GRAD".format(name)]
            temp_var += var * var_grad
        
        temp_var.backward()
        self.optimizer.step()
        self.layer.clear_gradients()

    def cancel(self):
        self.layer.clear_gradients()
        self.fetch_vars = None
