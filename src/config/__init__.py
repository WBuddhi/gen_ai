from src.config.spark_config import get_spark_session_db_client
import logging
import logging.config
import os
from typing import Dict

import yaml
from dotenv import load_dotenv
from yaml.loader import SafeLoader

spark, db_client = get_spark_session_db_client()

ENV = db_client.dbutils.widgets.get("env_prefix")
DB_RUN_ID = db_client.dbutils.widgets.get("job_id")
DB_JOB_ID = db_client.dbutils.widgets.get("run_id")
LOG_VARS = {"env": ENV, "db_run_id": DB_RUN_ID, "db_job_id": DB_JOB_ID}
LOGGER_PATH_FROM_ROOT = "src/config/logging.yaml"


def get_local_prod_config() -> Dict[str, str]:
    if "REPO_ROOT_PATH" in os.environ:
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
        cdim = LOG_VARS

        if "extra" not in kwargs:
            kwargs["extra"] = {}

        if "custom_dimensions" in kwargs["extra"]:
            cdim.update(kwargs["extra"]["custom_dimensions"])

        kwargs["extra"]["custom_dimensions"] = cdim
        return msg, kwargs


def get_logger(name):
    logger = logging.getLogger(name)
    return CustomDimensionsAdapter(logger)


logger_path, logger_name = get_local_prod_config()

with open(str(os.path.abspath(logger_path))) as file_handler:
    logging_config = yaml.load(file_handler, Loader=SafeLoader)
    logging.config.dictConfig(logging_config)

logger = get_logger(logger_name)
