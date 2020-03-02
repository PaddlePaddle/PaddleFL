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

import paddle.fluid as fluid
import numpy as np
import os


class Gru4rec_Reader:
    def __init__(self):
        pass

    def to_lodtensor(self, data, place):
        """ convert to LODtensor """
        seq_lens = [len(seq) for seq in data]
        cur_len = 0
        lod = [cur_len]
        for l in seq_lens:
            cur_len += l
            lod.append(cur_len)
        flattened_data = np.concatenate(data, axis=0).astype("int64")
        flattened_data = flattened_data.reshape([len(flattened_data), 1])
        res = fluid.LoDTensor()
        res.set(flattened_data, place)
        res.set_lod([lod])
        return res

    def lod_reader(self, reader, place):
        def feed_reader():
            for data in reader():
                lod_src_wordseq = self.to_lodtensor([dat[0] for dat in data],
                                                    place)
                lod_dst_wordseq = self.to_lodtensor([dat[1] for dat in data],
                                                    place)
                fe_data = {}
                fe_data["src_wordseq"] = lod_src_wordseq
                fe_data["dst_wordseq"] = lod_dst_wordseq
                yield fe_data

        return feed_reader

    def sort_batch(self, reader, batch_size, sort_group_size, drop_last=False):
        """
        Create a batched reader.
        """

        def batch_reader():
            r = reader()
            b = []
            for instance in r:
                b.append(instance)
                if len(b) == sort_group_size:
                    sortl = sorted(b, key=lambda x: len(x[0]), reverse=True)
                    b = []
                    c = []
                    for sort_i in sortl:
                        c.append(sort_i)
                        if (len(c) == batch_size):
                            yield c
                            c = []

            if drop_last == False and len(b) != 0:
                sortl = sorted(b, key=lambda x: len(x[0]), reverse=True)
                c = []
                for sort_i in sortl:
                    c.append(sort_i)
            if (len(c) == batch_size):
                yield c
                c = []

        # Batch size check
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integeral value, "
                "but got batch_size={}".format(batch_size))
        return batch_reader

    def reader_creator(self, file_dir):
        def reader():
            files = os.listdir(file_dir)
            for fi in files:
                with open(file_dir + '/' + fi, "r") as f:
                    for l in f:
                        l = l.strip().split()
                        l = [w for w in l]
                        src_seq = l[:len(l) - 1]
                        trg_seq = l[1:]
                        yield src_seq, trg_seq

        return reader

    def reader(self, file_dir, place, batch_size=5):
        """ prepare the English Pann Treebank (PTB) data """
        print("start constuct word dict")
        reader = self.sort_batch(
            self.reader_creator(file_dir), batch_size, batch_size * 20)
        return self.lod_reader(reader, place)
