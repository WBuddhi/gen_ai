import os
from datasets import load_dataset
from src.utils import load_yaml
from src.ingestion.utils.spark_utils import create_volume
from src.ingestion.transformations.base_transformer import BaseTransformer
import pyspark.pandas as pds
from src.config import logger


class ToBronze(BaseTransformer):
    def __init__(self, config_path: str, databricks_connect: str):
        self.config = load_yaml(config_path)
        super().__init__(
            **self.config["bronze"], databricks_connect=databricks_connect
        )
        self.dataset_name = self.config["dataset_name"]
        self.task = self.config["task"]

    def load_dataset(self):
        return load_dataset(self.dataset_name, self.task)

    def transform(self):
        dataset_cache_path = os.path.join(
            "/Volumes",
            self.catalog_name,
            "test_landing",
            self.task,
        )
        logger.info(f"Datasets cache dir: {dataset_cache_path}")
        create_volume(
            self.db_client, self.catalog_name, "test_landing", self.task
        )
        dataset = load_dataset(
            self.dataset_name, self.task, cache_dir=dataset_cache_path
        )
        dfs = []
        for dataset_name, data in dataset.items():
            dfs.append(
                {
                    "table_name": dataset_name,
                    "df": pds.DataFrame(data).to_spark(),
                }
            )
        return dfs
