from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from typing import Tuple, Dict
import os


def get_spark_session_db_client(
    name: str, databricks_connect: Dict[str, str]
) -> Tuple[SparkSession, WorkspaceClient]:
    if os.environ.get("DB_INSTANCE_TYPE", None):
        builder = SparkSession.builder.appName(name)
        db_client = WorkspaceClient()
    else:
        from databricks.connect import DatabricksSession
        from databricks.sdk.core import Config

        db_config = Config(**databricks_connect)
        builder = DatabricksSession.builder.sdkConfig(db_config)
        db_client = WorkspaceClient(**databricks_connect)
    return builder.getOrCreate(), db_client
