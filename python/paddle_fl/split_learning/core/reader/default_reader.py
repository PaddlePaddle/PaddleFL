import yaml
import ujson
import logging

from etl_pipeline import MapperExecutor
from feature_extractor import FeatureExtractor

from .reader_base import ReaderBase

class DefaultReader(ReaderBase):

    def __init__(self, etl_yaml_conf, fe_conf):
        super(DefaultReader, self).__init__()
        self.executor = self._init_etl_executor(etl_yaml_conf)
        self.fe = self._init_fe(fe_conf)

    def _init_etl_executor(self, etl_yaml_conf):
        executor = MapperExecutor()
        executor.build(etl_yaml_conf)
        return executor

    def _init_fe(self, fe_conf):
        fe = FeatureExtractor()
        fe.build(fe_conf)
        return fe

    def parse(self, db_value):
        if db_value is None:
            raise ValueError("db_value is None")
        group_json = self.executor.run_map_ops(db_value)
        # TODO: tobe removed, etl-pipeline py3 v0.3.1 not support
        group_json = [{
            "user_info": {
                "user_app_list": group_json[0]["user_app_list"],
                "uid": group_json[0]["uid"],
            },
            "item_info": [1],
        }]
        fea_dict = self.fe.extract_from_dict(group_json)
        return fea_dict
