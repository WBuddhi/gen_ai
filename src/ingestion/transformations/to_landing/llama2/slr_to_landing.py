import os
from datasets import load_datasets
from src.utils import load_yaml
from typing import Dict, List, Union
from src.ingestion.transformations.base_transformer import BaseTransformer
import pyspark.pandas as pds


class ToLanding(BaseTransformer):
    def __init__(self, dbutils, mode: str, config_path: str):
        self.config = load_yaml(config_path)
        self._set_class_vars(self.config)

    def _set_class_vars(config: Dict[str, Union[str, int, List, Dict]]):
        for key, value in config.items():
            setattr(key, value)

    def create_volumes(self, path: str, use_volumes: bool = False):
        if not use_volumes:
            return path

    def load_dataset(self):
        return load_datasets(self.dataset_name, self.dataset_subset)

    def transform(self):
        dataset_cache_path = os.path.join(
            "/Volumes",
            self.catalog_name,
            self.schema["landing"]["schema_name"],
            self.task,
        )
        self.create_volumes(
            dataset_cache_path, self.schema["landing"]["use_volumes"]
        )
        dataset = self.load_dataset(cache_dir=dataset_cache_path)
        dfs = {}
        for dataset_name, data in dataset.items():
            dfs[dataset_name] = pds.DataFrame(data).to_spark()
        return dfs
