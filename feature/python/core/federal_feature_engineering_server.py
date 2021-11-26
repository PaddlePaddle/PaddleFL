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
Federal feature engineering server-side interface
support postive_ratio, woe, iv, ks, auc
"""

import threading
import metrics_server as ms

class FederalFeatureEngineeringServer(object):
    """
    Federal feature engineering server-side implementation
    """
    def serve(self, server):
        """
        server init with grpc server
        """
        self._server = server
        
    def get_positive_ratio(self, features):
        """
        compute postive ratio with client
        prams:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
        """
        stop_event = threading.Event()
        ms.metrics_pb2_grpc.add_MpcPositiveRatioServicer_to_server(
                                    ms.MpcPositiveRatioServicer(features, stop_event), 
                                    self._server)
        self._server.start()
        stop_event.wait()
        self._server.stop(90)
        return {}

    def get_woe(self, features):
        """
        reutrn woe to server
        params:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
        return:
            a woe list including feature_size dicts
            each dict represents the woe (float) of each feature value
            e.g. [{1: 0.0, 0: 0.916291}, {2: -1.386294, 1: 0.0, 0: 0.916291}]    
        """
        woe_list = []
        stop_event = threading.Event()
        ms.metrics_pb2_grpc.add_MpcWOEServicer_to_server(
                                    ms.MpcWOEServicer(features, stop_event, woe_list), 
                                    self._server)
        self._server.start()
        stop_event.wait()
        self._server.stop(90)
        return woe_list
    
    def get_iv(self, features):
        """
        reutrn iv to server
        params:
            features: a feature list in the shape of (sample_size, features_size)
                      e.g. [[4, 3, 1], [1, 2, 5],...,[2, 3 ,2]] (feature_size = 3)
        return:
            an list corresponding to the iv of each feature
            e.g. [0.56653, 0.56653]    
        """
        iv_list = []
        stop_event = threading.Event()
        ms.metrics_pb2_grpc.add_MpcIVServicer_to_server(
                                    ms.MpcIVServicer(features, stop_event, iv_list), 
                                    self._server)
        self._server.start()
        stop_event.wait()
        self._server.stop(90)
        return iv_list