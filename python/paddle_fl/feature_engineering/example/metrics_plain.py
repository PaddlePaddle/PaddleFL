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
# limitations under the License
"""
get plain metrcis for test
"""

import math
import numpy as np

def get_plain_pos_ratio(labels, features):
    """
    get plain pos ratio for test
    """
    labels = [item for sublist in labels for item in sublist]
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    pos_ratio_list = []

    for feature_idx in range(len(features[0])):
        pos_sum = {}
        neg_sum = {}
        feature_bin = {}
        pos_ratio_dict = {}
        for sample_index in range(len(labels)):
            feature_value = features[sample_index][feature_idx]
            if(feature_value in feature_bin):
                pos_sum[feature_value] = pos_sum[feature_value] + labels[sample_index]
                feature_bin[feature_value] += 1
            else:
                pos_sum[feature_value] = labels[sample_index]
                feature_bin[feature_value] = 1
            
        for key, value in pos_sum.items():
            pos_ratio = float(pos_sum[key] / feature_bin[key])
            pos_ratio_dict[key] = round(pos_ratio, 6)
        pos_ratio_list.append(pos_ratio_dict)

    return pos_ratio_list


def get_plain_woe(labels, features):
    """
    get plain woe for test
    """
    labels = [item for sublist in labels for item in sublist]
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    woe_list = []

    for feature_idx in range(len(features[0])):
        pos_sum = {}
        neg_sum = {}
        feature_bin = {}
        woe_dict = {}
        for sample_index in range(len(labels)):
            feature_value = features[sample_index][feature_idx]
            if(feature_value in feature_bin):
                pos_sum[feature_value] = pos_sum[feature_value] + labels[sample_index]
                feature_bin[feature_value] += 1
            else:
                pos_sum[feature_value] = labels[sample_index]
                feature_bin[feature_value] = 1
            
        for key, value in pos_sum.items():
            neg_sum[key] = feature_bin[key] - pos_sum[key]
            woe = 0.0
            if pos_sum[key] == 0:
                woe = -20.0
            elif neg_sum[key] == 0:
                woe = 20.0
            else:
                woe = float(math.log((pos_sum[key] / total_pos) / (neg_sum[key] / total_neg)))
            woe_dict[key] = round(woe, 6)
        woe_list.append(woe_dict)
    return woe_list


def get_plain_iv(labels, features):
    """
    get plain iv for test
    """
    labels = [item for sublist in labels for item in sublist] 
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    iv_list = []
    for feature_idx in range(len(features[0])):
        pos_sum = {}
        neg_sum = {}
        feature_bin = {}
        for sample_index in range(len(labels)):
            feature_value = features[sample_index][feature_idx]
            if(feature_value in feature_bin):
                pos_sum[feature_value] = pos_sum[feature_value] + labels[sample_index]
                feature_bin[feature_value] += 1
            else:
                pos_sum[feature_value] = labels[sample_index]
                feature_bin[feature_value] = 1
            
        iv = 0
        for key, value in pos_sum.items():
            neg_sum[key] = feature_bin[key] - pos_sum[key]
            woe = 0.0
            if pos_sum[key] == 0:
                woe = -20.0
            elif neg_sum[key] == 0:
                woe = 20.0
            else:
                woe = float(math.log((pos_sum[key] / total_pos) / (neg_sum[key] / total_neg)))
            weight = (pos_sum[key] / total_pos) - (neg_sum[key] / total_neg)
            iv += weight * woe
        iv = round(iv, 6)
        iv_list.append(iv)
    
    return iv_list


def get_plain_ks(labels, features):
    """
    get plain ks for test
    """
    labels = [item for sublist in labels for item in sublist] 
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    ks_list = []
    for feature_idx in range(len(features[0])):
        pos_sum = {}
        neg_sum = {}
        feature_bin = {}
        for sample_index in range(len(labels)):
            feature_value = features[sample_index][feature_idx]
            if(feature_value in feature_bin):
                pos_sum[feature_value] = pos_sum[feature_value] + labels[sample_index]
                feature_bin[feature_value] += 1
            else:
                pos_sum[feature_value] = labels[sample_index]
                feature_bin[feature_value] = 1
        
        pos_sum = dict(sorted(pos_sum.items(), key = lambda item:item[0]))

        ks = -1
        cum_pos = 0
        cum_neg = 0
        for key, value in pos_sum.items():
            neg_sum[key] = feature_bin[key] - pos_sum[key]
            cum_pos += pos_sum[key]
            cum_neg += neg_sum[key]
            ks_temp = round(abs(cum_pos / total_pos - cum_neg / total_neg), 6)
            ks = max(ks_temp, ks)
        
        ks_list.append(ks)
    
    return ks_list


def get_plain_auc(labels, features):
    """
    get plain auc for test
    """
    labels = [item for sublist in labels for item in sublist] 
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    auc_list = []
    num_thresholds = 4095
    for feature_idx in range(len(features[0])):
        stat_pos_sum = {}
        stat_neg_sum = {}
        feature_bin = {}
        feature_values = [val[feature_idx] for val in features]
        Max = np.max(feature_values)
        Min = np.min(feature_values)
        feature_values = (feature_values - Min) / (Max - Min)
        for sample_index in range(len(labels)):
            bin_idx = feature_values[sample_index] * num_thresholds
            if(bin_idx in feature_bin):
                stat_pos_sum[bin_idx] = stat_pos_sum[bin_idx] + labels[sample_index]
                feature_bin[bin_idx] += 1
            else:
                stat_pos_sum[bin_idx] = labels[sample_index]
                feature_bin[bin_idx] = 1
        
        stat_pos_sum = dict(sorted(stat_pos_sum.items(), key = lambda item:item[0], reverse=True))

        tot_pos = 0.0
        tot_neg = 0.0
        auc = 0.0
        for key, value in stat_pos_sum.items():
            stat_neg_sum = feature_bin[key] - stat_pos_sum[key]
            tot_pos_prev = tot_pos
            tot_neg_prev = tot_neg
            tot_pos += stat_pos_sum[key]
            tot_neg += stat_neg_sum
            auc += abs((tot_neg - tot_neg_prev) * (tot_pos + tot_pos_prev) /2)
        
        auc = auc / tot_pos / tot_neg
        auc_list.append(round(auc, 6))
    
    return auc_list