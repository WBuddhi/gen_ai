import os
import re

from haystack.nodes import PreProcessor
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    collect_list,
    concat,
    concat_ws,
    lit,
    monotonically_increasing_id,
)
from transformers import LlamaTokenizer

from src.config import db_client, logger, spark
from src.ingestion.transformations.base_transformer import BaseTransformer
from src.utils import load_yaml, run_in_databricks


class GoldToFeatureStore(BaseTransformer):
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        super().__init__(
            **self.config["feature_store"], spark=spark, db_client=db_client
        )
        self.doc_preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by="sentence",
            split_length=1,
            split_respect_sentence_boundary=False,
            progress_bar=False,
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

    def create_feature_columns(self, df: DataFrame) -> DataFrame:
        data = df.select("doc_id", "article").collect()
        preprocessor_docs = [
            {
                "content": re.sub(
                    r'\s([?.!"](?:\s|$))',
                    r"\1",
                    re.sub(r"\([^)]*\)", "", item.article.strip()),
                ),
                "meta": {"doc_id": item.doc_id},
            }
            for item in data
        ]
        docs = self.doc_preprocessor.process(preprocessor_docs)
        split_docs = [
            {
                "doc_split_id": f"{doc.meta['doc_id']}_{doc.meta['_split_id']}",
                "article_snippet": doc.content,
                "doc_id": doc.meta["doc_id"],
                "snippet_len": len(doc.content.split()),
                "split_id": doc.meta["_split_id"],
            }
            for doc in docs
        ]
        df = spark.createDataFrame(split_docs)
        return df.dropDuplicates(["doc_split_id"])

    def clean_article(
        self,
        df: DataFrame,
        df_snippets: DataFrame,
        sensible_sentence_len_threshold: int = 8,
    ) -> DataFrame:
        logger.info("Creating clean article")
        df_clean = df_snippets.where(
            df_snippets["snippet_len"] > sensible_sentence_len_threshold
        )
        window = (
            Window.partitionBy("doc_id")
            .orderBy("split_id")
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )

        df_clean = df_clean.select(
            "doc_id",
            collect_list(col("article_snippet"))
            .over(window)
            .alias("snippet_array"),
        ).dropDuplicates(["doc_id"])
        df_clean = df_clean.select(
            col("doc_id").alias("doc_id_clean"),
            concat_ws(" ", col("snippet_array")).alias("article_clean"),
        )
        return df.join(
            df_clean, df["doc_id"] == df_clean["doc_id_clean"], "inner"
        ).drop(col("doc_id_clean"))

    def transform(self):
        dfs = []
        tokenizer = LlamaTokenizer.from_pretrained(
            self.config["model"], legacy=False
        )
        for table_name, df in self.load_dataset().items():
            logger.info(f"Processing: {table_name} df")
            df = df.withColumn("doc_id", monotonically_increasing_id())
            df_snippets = self.create_feature_columns(df)
            self.cache_table(df_snippets)
            df = self.clean_article(df, df_snippets)
            logger.info("Adding prompt")
            df = df.withColumn(
                "model_input",
                concat(
                    lit("<s>[INS] "),
                    col("prompt"),
                    lit("\nARTICLE:\n"),
                    col("article_clean"),
                    lit("\nPlain Language Summary:[INST]\n"),
                    col("summary"),
                    lit(tokenizer.eos_token),
                ),
            )
            df_snippets = {
                "table_name": f"{table_name}_snippets",
                "df": df_snippets,
                "primary_keys": "doc_split_id",
            }
            df = {
                "table_name": table_name,
                "df": df,
                "primary_keys": "doc_id",
            }
            dfs.extend([df_snippets, df])
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
