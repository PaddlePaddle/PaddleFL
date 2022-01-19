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
Federated feature engineering client-side
support postive_ratio, woe, iv, ks, auc
"""
import logging
from . import metrics_client as mc

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FederatedFeatureEngineeringClient(object):
    """
    Federated feature engineering client-side implementation
    """
    def __init__(self, key_len=1024):
        """
        init paillier instance with given key_len 
        """
        self._paillier = mc.hu.Paillier()
        self._paillier.keygen(key_len)
        logger.info('keygen done, key_len is {} bits'.format(key_len))

    def connect(self, channel):
        """
        client init the grpc channel with server
        """
        self._channel = channel

    def get_positive_ratio(self, labels):
        """
        return postive ratio to client
        params:
            labels: a list in the shape of (sample_size, 1)
                    labels[i] is either 0 or 1, represents negative and positive resp.
                    e.g. [[1], [0], [1],...,[1]]
        return:
            a positive ratio list including feature_size dicts
            each dict represents the positive ratio (float) of each feature value
            e.g. [{0: 0.2, 1: 0.090909}, {1: 0.090909, 0: 0.2, 2: 0.02439}...]    
        """
        return mc.get_mpc_postive_ratio_alice(self._channel, labels, self._paillier)

    def get_woe(self, labels):
        """
        return woe to client
        params:
            labels: a list in the shape of (sample_size, 1)
                    labels[i] is either 0 or 1, represents negative and positive resp.
                    e.g. [[1], [0], [1],...,[1]]
        return:
            a woe list including feature_size dicts
            each dict represents the woe (float) of each feature value
            e.g. [{1: 0.0, 0: 0.916291}, {2: -1.386294, 1: 0.0, 0: 0.916291}]    
        """
        return mc.get_mpc_woe_alice(self._channel, labels, self._paillier)

    def get_iv(self, labels):
        """
        return iv to client
        params:
            labels: a list in the shape of (sample_size, 1)
                    labels[i] is either 0 or 1, represents negative and positive resp.
                    e.g. [[1], [0], [1],...,[1]]
        return:
            a list corresponding to the iv of each feature
            e.g. [0.56653, 0.56653]
        """
        return mc.get_mpc_iv_alice(self._channel, labels, self._paillier)

    def get_woe_iv(self, labels):
        """
        return woe, iv to client
        params:
            labels: a list in the shape of (sample_size, 1)
                    labels[i] is either 0 or 1, represents negative and positive resp.
                    e.g. [[1], [0], [1],...,[1]]
        return:
            a tuple of woe and iv
        """
        return mc.get_mpc_iv_alice(self._channel, labels, self._paillier, True)
    
    def get_ks(self, labels):
        """
        reutrn ks to client
        params:
            labels: a list in the shape of (sample_size, 1)
                    labels[i] is either 0 or 1, represents negative and positive resp.
                    e.g. [[1], [0], [1],...,[1]]
        return:
            a list corresponding to the ks of each feature
            e.g. [0.3, 0.3]
        """
        return mc.get_mpc_ks_alice(self._channel, labels, self._paillier)

    def get_auc(self, labels):
        """
        reutrn auc to client
        params:
            labels: a list in the shape of (sample_size, 1)
                    labels[i] is either 0 or 1, represents negative and positive resp.
                    e.g. [[1], [0], [1],...,[1]]
        return:
            a list corresponding to the auc of each feature
            e.g. [0.33, 0.33]
        """
        return mc.get_mpc_auc_alice(self._channel, labels, self._paillier)
