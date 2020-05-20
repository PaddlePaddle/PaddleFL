# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle.fluid as fluid


class LayerBase(object):
    def __init__(self):
        pass


class Embedding(LayerBase):
    def __init__(self):
        pass

    def attr(self, shapes, name=None):
        self.vocab_size = shapes[0]
        self.emb_dim = shapes[1]
        self.name = name

    def forward(self, i):
        is_sparse = True
        if self.name is not None:
            param_attr = fluid.ParamAttr(name=self.name)
        else:
            param_attr = None

        results = []
        emb = fluid.layers.embedding(
            input=i,
            is_sparse=is_sparse,
            size=[self.vocab_size, self.emb_dim],
            param_attr=param_attr,
            padding_idx=0)
        return emb


class SequenceConv(LayerBase):
    def __init__(self):
        pass

    def attr(self, name=None):
        self.num_filters = 64
        self.win_size = 3
        self.name = name

    def forward(self, i):
        if self.name is not None:
            param_attr = fluid.ParamAttr(name=self.name)
        else:
            param_attr = None
        conv = fluid.nets.sequence_conv_pool(
            input=i,
            num_filters=self.num_filters,
            filter_size=self.win_size,
            param_attr=param_attr,
            act="tanh",
            pool_type="max")
        return conv


class Concat(LayerBase):
    def forward(self, inputs):
        concat = fluid.layers.concat(inputs, axis=1)
        return concat


class Pooling(LayerBase):
    def __init__(self):
        self.pool_type = 'sum'

    def attr(self, pool_type='sum'):
        self.pool_type = pool_type

    def forward(self, i):
        pool = fluid.layers.sequence_pool(input=i, pool_type='sum')
        return pool


class FC(LayerBase):
    def __init__(self):
        return

    def attr(self, shapes, act='relu', name=None):
        self.name = name
        self.size = shapes[0]
        self.act = act

    def forward(self, i):
        if self.name is not None:
            param_attr = fluid.ParamAttr(name=self.name)
        else:
            param_attr = None
        fc = fluid.layers.fc(input=i,
                             size=self.size,
                             act=self.act,
                             param_attr=param_attr)
        return fc


class CrossEntropySum(LayerBase):
    def __init__(self):
        pass

    def forward(self, prediction, label):
        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        sum_cost = fluid.layers.reduce_sum(cost)
        return sum_cost


class ACC(LayerBase):
    def __init__(self):
        pass

    def forward(self, prediction, label):
        return fluid.layers.accuracy(input=prediction, label=label)


class AUC(LayerBase):
    def forward(self, prediction, label):
        auc, batch_auc_var, auc_states = \
                    fluid.layers.auc(input=prediction, label=label,
                                     num_thresholds=2 ** 12,
                                     slide_steps=20)
        return auc, batch_auc_var
