from src.ingestion.transformations.to_bronze.llama2.source_to_bronze import (
    SourceToBronze,
)

source_to_bronze = SourceToBronze(
    config_path="../../pipeline_configs/llama2_7b_32k_slr.yaml"
)
source_to_bronze.run()
