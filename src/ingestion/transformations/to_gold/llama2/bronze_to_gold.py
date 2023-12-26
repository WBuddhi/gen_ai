import os
from src.config import logger, spark, db_client
from src.ingestion.transformations.base_transformer import BaseTransformer
from src.utils import load_yaml, run_in_databricks
from pyspark.sql.functions import lit


class BronzeToGold(BaseTransformer):
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        super().__init__(
            **self.config["gold"], spark=spark, db_client=db_client
        )

    def load_dataset(self):
        logger.info("Loading dataframs from Bronze layer")
        full_schema_name = f"{self.catalog_name}.bronze"
        dfs = {}
        for table_name in ("train", "validation", "test"):
            dfs[table_name] = self.spark.table(
                f"{full_schema_name}.{table_name}"
            )
        return dfs

    def transform(self):
        prompt = "Create a Plain Language Summary of the following article:"
        dfs = []
        logger.info("Adding prompt column")
        for table_name, df in self.load_dataset().items():
            df_data = {
                "table_name": table_name,
                "df": df.withColumn("prompt", lit(prompt)),
            }
            dfs.append(df_data)
        return dfs


if __name__ == "__main__":
    config_relative_path = "src/pipeline_configs/llama2_7b_pls.yaml"
    config_path = (
        os.path.join(os.environ["REPO_ROOT_PATH"], config_relative_path)
        if run_in_databricks()
        else os.path.join(".", config_relative_path)
    )
    bronze_to_gold = BronzeToGold(config_path=config_path)
    bronze_to_gold.run()
