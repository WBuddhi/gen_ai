from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from typing import Tuple
import os


def get_spark_session(
    name: str, config_path: str = None
) -> Tuple[SparkSession, WorkspaceClient]:
    if os.environ.get("DB_INSTANCE_TYPE", None):
        builder = SparkSession.builder.appName(name)
        db_client = WorkspaceClient()
    else:
        from databricks.connect import DatabricksSession
        from databricks.sdk.core import Config
        import yaml

        with open(config_path, "r") as yaml_stream:
            config = yaml.safe_load(yaml_stream)
        db_config = Config(**config)
        builder = DatabricksSession.builder.sdkConfig(db_config)
        db_client = WorkspaceClient(db_config)
    return builder.getOrCreate()
