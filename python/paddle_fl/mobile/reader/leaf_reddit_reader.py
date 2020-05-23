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

import collections
import os
import sys
import numpy as np
from utils.logger import logging
import json

PAD_SYMBOL, UNK_SYMBOL = 0, 1
DATA_PATH = "lm_data"
VOCAB_PATH = os.path.join(DATA_PATH, "vocab.json")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "20200101", "train_data.json")
VOCAB = None


def build_counter(train_data):
    train_tokens = []
    for u in train_data:
        for c in train_data[u]['x']:
            train_tokens.extend([s for s in c])

    all_tokens = []
    for i in train_tokens:
        all_tokens.extend(i)
    train_tokens = []

    counter = collections.Counter()
    counter.update(all_tokens)
    all_tokens = []
    return counter


def build_vocab(filename, vocab_size=10000):
    train_data = {}
    with open(filename) as json_file:
        data = json.load(json_file)
        train_data = data['user_data']
    counter = build_counter(train_data)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # -2 to account for the unknown and pad symbols
    count_pairs = count_pairs[:(vocab_size - 2)]
    words, _ = list(zip(*count_pairs))

    vocab = {}
    vocab['<PAD>'] = PAD_SYMBOL
    vocab['<UNK>'] = UNK_SYMBOL

    for i, w in enumerate(words):
        if w != '<PAD>':
            vocab[w] = i + 1

    return {
        'vocab': vocab,
        'size': vocab_size,
        'unk_symbol': vocab['<UNK>'],
        'pad_symbol': vocab['<PAD>']
    }


def save_vocab(filename, vocab):
    with open(filename, "w") as f:
        f.write(json.dumps(vocab))


def load_vocab(filename):
    with open(filename) as f:
        return json.loads(f.read())


if os.path.exists(VOCAB_PATH):
    logging.info("load vocab form: {}".format(VOCAB_PATH))
    VOCAB = load_vocab(VOCAB_PATH)
else:
    #TODO: singleton
    logging.info("build vocab form: {}".format(TRAIN_DATA_PATH))
    VOCAB = build_vocab(TRAIN_DATA_PATH)
    logging.info("save vocab into: {}".format(VOCAB_PATH))
    save_vocab(VOCAB_PATH, VOCAB)
if VOCAB is None:
    logging.error("load vocab error")
    raise Exception("load vocab error")


def train_reader(lines):
    def local_iter():
        seg_id = 0
        for line in lines:
            assert (len(line.split("\t")) == 3)
            uid, _, input_str = line.split("\t")
            data = json.loads(input_str)
            data_x = data["x"]
            data_y = data["y"]
            data_mask = data["mask"]

            input_data, input_length = process_x(data_x, VOCAB)
            target_data = process_y(data_y, VOCAB)
            yield [input_data] + [target_data] + [input_length] + [data_mask]

    return local_iter


def infer_reader(lines):
    return train_reader(lines)


def load_data_into_patch(filelist, patch_size):
    data_patch = []
    idx = 0
    local_user_dict = {}
    for fn in filelist:
        tmp_list = []
        with open(fn) as fin:
            raw_data = json.loads(fin.read())["user_data"]
        local_user_dict = {k: 0 for k in raw_data.keys()}
        for user, data in raw_data.items():
            data_x = data["x"]
            data_y = data["y"]
            for c, l in zip(data_x, data_y):
                for inst_i in range(len(c)):
                    local_user_dict[user] += 1
                    idx += 1
                    inst = {
                        "x": c[inst_i],
                        "y": l["target_tokens"][inst_i],
                        "mask": l["count_tokens"][inst_i]
                    }
                    line = "{}\t\t{}".format(user, json.dumps(inst))
                    if idx % patch_size == 0:
                        data_patch.append(tmp_list)
                        tmp_list = [line]
                    else:
                        tmp_list.append(line)
        if len(tmp_list) > 0:
            data_patch.append(tmp_list)
    return data_patch, local_user_dict


def tokens_to_ids(tokens, vocab):
    to_ret = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    return np.array(to_ret, dtype="int64")


def process_x(raw_x, vocab):
    tokens = tokens_to_ids(raw_x, vocab["vocab"])
    lengths = np.sum(tokens != vocab["pad_symbol"])
    return tokens, lengths


def process_y(raw_y, vocab):
    tokens = tokens_to_ids(raw_y, vocab["vocab"])
    return tokens
