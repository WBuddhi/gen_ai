from src.config import logger, db_client
from src.ingestion.transformations.to_bronze.llama2.source_to_bronze import (
    SourceToBronze,
)

some_input = db_client.dbutils.widgets.get["some_input"]
logger.info(f"Input parameter: {some_input}")

source_to_bronze = SourceToBronze(config_path = "../../pipeline_configs/llama2_7b_32k_slr.yaml")
source_to_bronze.run()
