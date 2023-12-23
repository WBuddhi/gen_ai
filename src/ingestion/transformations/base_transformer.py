from abc import ABCMeta, abstractmethod
from typing import List

from pyspark.sql import DataFrame

from src.config import logger
from src.ingestion.utils.spark_utils import save
from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureEngineeringClient


class BaseTransformer(metaclass=ABCMeta):
    def __init__(
        self,
        task_name: str,
        catalog_name: str,
        schema_name: str,
        mode: str,
        spark: SparkSession,
        db_client: DatabricksSession,
        destination_file_format: str = "DELTA",
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.destination_file_format = destination_file_format
        self.task_name = task_name
        self.spark = spark
        self.db_client = db_client
        self.mode = mode
        self.cached_tables = []

    @abstractmethod
    def transform(self) -> List[DataFrame]:
        raise NotImplementedError

    def cache_table(self, df: DataFrame):
        """Cache tables and appends them to list.

        This method caches a dataframe and appends them to a list.
        All dataframes in the list are unpersisted upon run completion.

        Args:
            df (DataFrame): df
        """
        logger.info("Caching dataframe")
        df = df.cache()
        self.cached_tables.append(df)

    def _unpersist_cache(self):
        """Unpersists all cached tables

        This method unpersists all dataframes cached using the
        `cache_table` method. Method called at the end of the run.
        """
        for df in self.cached_tables:
            df.unpersist()

    def run(
        self,
        optimize_table: bool = False,
    ) -> None:
        fe_client = (
            FeatureEngineeringClient()
            if self.destination_file_format == "FEATURESTORE"
            else None
        )
        try:
            dfs = self.transform()
        except Exception as error:
            logger.exception("Failed to apply transformation")
            raise error

        for df in dfs:
            try:
                save(
                    spark=self.spark,
                    db_client=self.db_client,
                    df=df["df"],
                    catalog_name=self.catalog_name,
                    schema_name=self.schema_name,
                    table_name=df["table_name"],
                    file_format=self.destination_file_format,
                    mode=self.mode,
                    partition_by=df.get("partition_by", None),
                    primary_keys=df.get("primary_keys", None),
                    optimize_table=optimize_table,
                    upsert_config=df.get("upsert_config", None),
                    feature_store_client=fe_client,
                )
            except Exception as error:
                logger.exception("Failed to save tables")
                raise error
        self._unpersist_cache()
