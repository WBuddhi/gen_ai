import os
from src.config import logger, spark, db_client
from src.ingestion.transformations.base_transformer import BaseTransformer
from src.utils import load_yaml, run_in_databricks
from pyspark.sql.functions import monotonically_increasing_id, concat, lit, col
from transformers import LlamaTokenizer


class GoldToFeatureStore(BaseTransformer):
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        super().__init__(
            **self.config["feature_store"], spark=spark, db_client=db_client
        )

    def load_dataset(self):
        logger.info("Loading dataframs from Bronze layer")
        full_schema_name = f"{self.config['gold']['catalog_name']}.gold"
        dfs = {}
        for table_name in ("train", "validation", "test"):
            dfs[table_name] = self.spark.table(
                f"{full_schema_name}.{table_name}"
            )
        return dfs

    def transform(self):
        dfs = []
        tokenizer = LlamaTokenizer.from_pretrained(self.config["model"])
        tokenizer.pad_token = "<PAD>"
        tokenizer.padding_side = "right"
        for table_name, df in self.load_dataset().items():
            columns = [col(column_name) for column_name in df.columns]
            columns.append(
                concat(
                    lit("<s>[INS] "),
                    col("prompt"),
                    lit(": "),
                    col("article"),
                    lit(" [INST]PLS: "),
                    col("summary"),
                    lit(tokenizer.eos_token),
                ).alias("model_input")
            )
            df = df.select(*columns)
            df = df.withColumn("id", monotonically_increasing_id())
            df_data = {
                "table_name": table_name,
                "df": df,
                "description": f"{table_name} table for LLama2 Finetuning for PLS task",
                "primary_keys": "id",
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
    gold_to_fs = GoldToFeatureStore(config_path=config_path)
    gold_to_fs.run()
