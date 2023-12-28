import os
from typing import Dict, Tuple

import torch
from datasets import IterableDataset
from mlflow import (
    autolog,
    create_experiment,
    get_experiment_by_name,
    set_experiment,
    start_run,
)
from mlflow.exceptions import MlflowException
from peft import LoraConfig
from pyspark.sql.functions import col
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from src.config import db_client, logger, spark
from src.utils import load_yaml, run_in_databricks


class PLSTrainer:
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        self.__dict__.update(self.config["feature_store"])
        self.__dict__.update(self.config["modelling"])
        self.experiment_id = self.mlflow_setup()
        autolog()

    def mlflow_setup(self):
        try:
            experiment_id = create_experiment(self.experiment_name)
        except MlflowException:
            logger.info("Experiment already exists")
            experiment_id = get_experiment_by_name(
                self.experiment_name
            ).experiment_id
        set_experiment(experiment_id=experiment_id)

    def load_dataset(self) -> Dict[str, IterableDataset]:
        logger.info("Loading data from feature store")
        full_schema_name = f"{self.catalog_name}.{self.schema_name}"
        datasets = {}
        for table_name in ("train", "validation", "test"):
            datasets[table_name] = IterableDataset.from_spark(
                spark.table(f"{full_schema_name}.{table_name}").select(
                    col("model_input")
                )
            )
        return datasets

    def setup_hf_mlflow(self):
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = 1

    def model(self) -> Tuple(LlamaForCausalLM, LlamaTokenizer):
        self.bits_and_bytes_config["bnb_4bit_compute_dtype"] = getattr(
            torch, self.bits_and_bytes_config["bnb_4bit_compute_dtype"]
        )
        quant_config = BitsAndBytesConfig(**self.bits_and_bytes_config)
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map={"": 0},
            use_flash_attention_2=self.use_flash_attention_2,
        )
        model.config.use_cache = self.model_config["use_cache"]
        model.config.pretraining_tp = self.model_config["pretraining_tp"]

        tokenizer = LlamaTokenizer.from_pretrained(
            self.base_model, legacy=False
        )
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def trainer(self) -> SFTTrainer:
        peft_config = LoraConfig(**self.lora_config)
        training_args = TrainingArguments(**self.training_params)
        model, tokenizer = self.model()
        dataset = self.load_dataset()

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            peft_config=peft_config,
            args=training_args,
            tokenizer=tokenizer,
            **self.sft_trainer_config,
        )
        return trainer

    def run(self):
        with start_run():
            trainer = self.trainer()
            trainer.train()


if __name__ == "__main__":
    config_relative_path = "src/pipeline_configs/llama2_7b_pls.yaml"
    config_path = (
        os.path.join(os.environ["REPO_ROOT_PATH"], config_relative_path)
        if run_in_databricks()
        else os.path.join(".", config_relative_path)
    )
    pls_trainer = PLSTrainer(config_path=config_path)
    pls_trainer.run()
