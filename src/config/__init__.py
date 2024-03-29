import logging.config
import os
import sys

import yaml
from yaml.loader import SafeLoader

from src.config.utils import (
    get_local_prod_config,
    get_logger,
    get_spark_session_db_client,
)
from src.utils import get_path_to_src

spark, db_client = get_spark_session_db_client()

if len(sys.argv) >= 4:
    ENV = sys.argv[1]
    DB_RUN_ID = sys.argv[2]
    DB_JOB_ID = sys.argv[3]
else:
    ENV = "Local"
    DB_RUN_ID = "local_run"
    DB_JOB_ID = "local_job"
LOG_VARS = {"env": ENV, "db_run_id": DB_RUN_ID, "db_job_id": DB_JOB_ID}


os.environ["REPO_ROOT_PATH"] = get_path_to_src()
logger_path, logger_name = get_local_prod_config()

with open(logger_path) as file_handler:
    logging_config = yaml.load(file_handler, Loader=SafeLoader)
    logging.config.dictConfig(logging_config)

logger = get_logger(logger_name, LOG_VARS)
logger.info(
    f"Inputs parameters: Env: {ENV}, DB Job id: {DB_JOB_ID}, DB Run id: {DB_RUN_ID}"
)
