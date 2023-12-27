import logging
import os
from typing import Dict, Tuple

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from pyspark.sql import SparkSession

from src.utils import load_yaml, run_in_databricks

LOGGER_PATH_FROM_ROOT = "src/config/logging.yaml"


def get_spark_session_db_client() -> Tuple[SparkSession, WorkspaceClient]:
    if run_in_databricks():
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


def get_local_prod_config() -> Dict[str, str]:
    if run_in_databricks():
        logger_path = os.path.join(
            os.environ["REPO_ROOT_PATH"], LOGGER_PATH_FROM_ROOT
        )
        logger_name = "defaultLogger"
    else:
        load_dotenv()
        logger_path = os.path.join(".", LOGGER_PATH_FROM_ROOT)
        logger_name = "defaultLogger"
    return logger_path, logger_name


class CustomDimensionsAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        cdim = {}

        if "extra" not in kwargs:
            kwargs["extra"] = {}

        if "custom_dimensions" in kwargs["extra"]:
            cdim.update(kwargs["extra"]["custom_dimensions"])

        kwargs["extra"]["custom_dimensions"] = cdim
        return msg, kwargs


def get_logger(name, custom_dimensions: Dict[str, str]):
    logger = logging.getLogger(name)
    return CustomDimensionsAdapter(
        logger, {"custom_dimensions": custom_dimensions}
    )
