import os
from datasets import load_dataset
from src.utils import load_yaml, run_in_databricks
from src.ingestion.utils.spark_utils import create_volume
from src.ingestion.transformations.base_transformer import BaseTransformer
import pyspark.pandas as pds
from src.config import logger, spark, db_client


class SourceToBronze(BaseTransformer):
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        super().__init__(
            **self.config["bronze"], spark=spark, db_client=db_client
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


if __name__ == "__main__":
    config_path = (
        "../../pipeline_configs/llama2_7b_32k_slr.yaml"
        if run_in_databricks()
        else "./src/pipeline_configs/llama2_7b_32k_slr.yaml"
    )
    source_to_bronze = SourceToBronze(config_path=config_path)
    source_to_bronze.run()
