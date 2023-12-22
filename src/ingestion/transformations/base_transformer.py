from abc import ABCMeta, abstractmethod
from typing import List, Dict

from pyspark.sql import DataFrame

from src.config import logger
from src.ingestion.spark_config import get_spark_session_db_client
from src.ingestion.utils.spark_utils import save


class BaseTransformer(metaclass=ABCMeta):
    def __init__(
        self,
        task_name: str,
        catalog_name: str,
        schema_name: str,
        mode: str,
        databricks_connect: Dict[str, str],
        destination_file_format: str = "DELTA",
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.destination_file_format = destination_file_format
        self.task_name = task_name
        self.spark, self.db_client = get_spark_session_db_client(
            self.task_name, databricks_connect
        )
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
        partition_by=None,
        optimize_table: bool = False,
    ) -> None:
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
                    partition_by=partition_by,
                    optimize_table=optimize_table,
                    upsert_config=df.get("upsert_config", None),
                )
            except Exception as error:
                logger.exception("Failed to save tables")
                raise error
        self._unpersist_cache()
