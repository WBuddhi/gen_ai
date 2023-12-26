import os
from src.config import logger, spark, db_client
from src.ingestion.transformations.base_transformer import BaseTransformer
from src.utils import load_yaml, run_in_databricks
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    monotonically_increasing_id,
    concat,
    lit,
    col,
    concat_ws,
)
from transformers import LlamaTokenizer
import spacy
import re


class GoldToFeatureStore(BaseTransformer):
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        super().__init__(
            **self.config["feature_store"], spark=spark, db_client=db_client
        )
        self.nlp = spacy.load("en_core_web_trf")

    def load_dataset(self):
        logger.info("Loading dataframs from Bronze layer")
        full_schema_name = f"{self.config['gold']['catalog_name']}.gold"
        dfs = {}
        for table_name in ("train", "validation", "test"):
            dfs[table_name] = self.spark.table(
                f"{full_schema_name}.{table_name}"
            )
        return dfs

    def create_feature_columns(self, df: DataFrame) -> DataFrame:
        rows = df.select("id", "article").collect()
        data = []
        for row in rows:
            article_id = row.id
            article = row.article
            snippets = []
            docs = self.nlp(article)
            snippets = [
                re.sub(
                    r'\s([?.!"](?:\s|$))',
                    r"\1",
                    re.sub(r"\([^)]*\)", "", sent.text.strip()),
                )
                for sent in docs
            ]
            article_data = [
                {"doc_id": article_id, "snippet_id": cnt, "snippet": snippet}
                for cnt, snippet in zip(range(0, len(snippets) + 1), snippets)
            ]
            data.extend(article_data)
        return self.spark.createDataFrame(data)

    def transform(self):
        dfs = []
        tokenizer = LlamaTokenizer.from_pretrained(
            self.config["model"], legacy=False
        )
        for table_name, df in self.load_dataset().items():
            df = df.withColumn("id", monotonically_increasing_id())
            df_snippets = self.create_feature_columns(df)
            df = df.join(df_snippets, df["id"] == df["doc_id"], "right").drop(
                ["id"]
            )
            df_clean = df.groupBy("doc_id").agg(
                concat_ws(" ", col("snippet")).alias("article_clean")
            )
            df = df.join(df_clean, df["doc_id"] == df_clean["doc_id"], "right")
            columns = [col(column_name) for column_name in df.columns]
            columns.append(
                concat(
                    lit("<s>[INS] "),
                    col("prompt"),
                    lit("\nARTICLE:\n"),
                    col("article_clean"),
                    lit("\nPlain Language Summary:[INST]\n"),
                    col("summary"),
                    lit(tokenizer.eos_token),
                ).alias("model_input")
            )
            df = df.select(*columns)
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
