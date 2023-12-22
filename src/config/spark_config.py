from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.connect import DatabricksSession
from src.utils import load_yaml
from typing import Tuple
import os


def get_spark_session_db_client() -> Tuple[SparkSession, WorkspaceClient]:
    if os.environ.get("LOGGER_PATH_FROM_ROOT", None):
        builder = SparkSession.builder
    else:
        databricks_connect_config = load_yaml(
            "./deploy/env_configs/local.yaml"
        )
        databricks_profile_name = databricks_connect_config[
            "databricks_connect"
        ]["profile"]
        builder = DatabricksSession.builder.profile(databricks_profile_name)
    db_client = WorkspaceClient()
    return builder.getOrCreate(), db_client
