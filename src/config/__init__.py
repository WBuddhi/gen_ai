import logging
import logging.config
import os
from typing import Dict

import yaml
from dotenv import load_dotenv
from yaml.loader import SafeLoader

ENV = os.environ.get("ENV", "dev")
DB_RUN_ID = os.environ.get("DB_RUN_ID", "dummy_run_id")
DB_JOB_ID = os.environ.get("DB_JOB_ID", "dummy_job_id")
LOG_VARS = {"env": ENV, "db_run_id": DB_RUN_ID, "db_job_id": DB_JOB_ID}


def get_local_prod_config() -> Dict[str, str]:
    if "PRICE_ELASTICITY_CONFIG_FILE_PATH" in os.environ:
        logger_path = os.path.join(
            os.environ["PRICE_ELASTICITY_CONFIG_FILE_PATH"], "logging.yaml"
        )
        logger_name = "defaultLogger"
    else:
        load_dotenv()
        logger_path = "/Workspace/Repos/buddhika.de.seram+databricks@blend360.com/gen_ai/src/config/logging.yaml"
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
