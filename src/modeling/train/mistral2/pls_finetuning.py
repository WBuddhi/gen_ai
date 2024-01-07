import os
from typing import Dict, Tuple

import torch
import pandas as pd
from datasets import Dataset
from mlflow import (
    MlflowException,
    create_experiment,
    enable_system_metrics_logging,
    get_experiment_by_name,
    set_experiment,
    start_run,
    pyfunc,
)
from peft import LoraConfig, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from src.config import logger, spark
from src.modelling.model.llm_qlora import LlmQlora
from src.utils import load_yaml, run_in_databricks
from huggingface_hub import snapshot_download
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, Schema


class PLSTrainer:
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        self.__dict__.update(self.config["feature_store"])
        self.__dict__.update(self.config["modelling"])
        self.__dict__.update(self.config["inference"])
        self.experiment_id = self.mlflow_setup()
        enable_system_metrics_logging()

    def mlflow_setup(self):
        try:
            experiment_id = create_experiment(self.experiment_name)
        except MlflowException:
            logger.info("Experiment already exists")
            experiment_id = get_experiment_by_name(
                self.experiment_name
            ).experiment_id
        set_experiment(experiment_id=experiment_id)

    def format_prompt(
        example: Dict[str, str], training: bool = True
    ) -> Dict[str, str]:
        instruction = (
            "Create a plain language summary for this scientific article"
        )
        system_inst = "You create laymen summaries of highly technical articles created by the biomedical industry"
        system_prompt = "{system_instruction}".format(
            system_instruction="{system_inst}"
        )
        instruction_prompt = """[INST]User: {instruction}
        #ARTICLE:
        {article}
        [/INST]""".format(
            article="{article}", instruction="{instruction}"
        )
        article = example.get("gpt_summary")
        response = example.get("summary") if training else ""
        full_prompt = "\n".join(
            [
                system_prompt.format(system_prompt=system_prompt),
                instruction_prompt.format(article=article),
                response,
            ]
        )
        return {"text": full_prompt}

    def load_dataset(self) -> Dict[str, Dataset]:
        logger.info("Loading data from feature store")
        full_schema_name = f"{self.catalog_name}.{self.schema_name}"
        datasets = {}
        for training_task in ("train", "validation", "test"):
            table_name = f"{training_task}_gpt_summary"
            try:
                df = spark.table(f"{full_schema_name}.{table_name}").select(
                    "gpt_summary"
                )
                datasets[training_task] = Dataset.from_spark(df)
            except Exception as _:
                logger.warn(f"Error loading table: {table_name}")
        return datasets

    def model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.bits_and_bytes_config["bnb_4bit_compute_dtype"] = getattr(
            torch, self.bits_and_bytes_config["bnb_4bit_compute_dtype"]
        )
        quant_config = BitsAndBytesConfig(**self.bits_and_bytes_config)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map={"": 0},
            use_flash_attention_2=self.use_flash_attention_2,
            cache_dir=self.hf_cache_dir,
        )
        model.config.use_cache = self.model_config["use_cache"]
        model.config.pretraining_tp = self.model_config["pretraining_tp"]

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            legacy=False,
            padding_side=self.padding_side,
            cache_dir=self.hf_cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def trainer(self) -> SFTTrainer:
        model, tokenizer = self.model()
        peft_config = LoraConfig(**self.lora_config)
        training_args = TrainingArguments(
            **self.training_params,
        )
        self.dataset = self.load_dataset()
        dataset = self.dataset.map(self.format_prompt)
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

            model_save_path = self.model_save_path
            trainer.save_model(model_save_path)

            peft_model_id = model_save_path
            config = PeftConfig.from_pretrained(peft_model_id)

            snapshot_location = snapshot_download(
                repo_id=config.base_model_name_or_path
            )

            input_schema = Schema(
                [
                    ColSpec(DataType.string, "prompt"),
                    ColSpec(DataType.double, "temperature"),
                    ColSpec(DataType.long, "max_tokens"),
                ]
            )
            output_schema = Schema([ColSpec(DataType.string)])
            signature = ModelSignature(
                inputs=input_schema, outputs=output_schema
            )

            prompt = self.format_prompt(self.dataset[0], training=False)
            temperature = self.input_example["temperature"]
            max_tokens = self.input_example["max_tokens"]
            input_example = pd.DataFrame(
                {
                    "prompt": [prompt],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

            pyfunc.log_model(
                **self.model_package,
                python_model=LlmQlora(),
                artifacts={
                    "repository": snapshot_location,
                    "lora": peft_model_id,
                },
                input_example=input_example,
                signature=signature,
            )


if __name__ == "__main__":
    config_relative_path = "src/pipeline_configs/llama2_7b_pls.yaml"
    config_path = (
        os.path.join(os.environ["REPO_ROOT_PATH"], config_relative_path)
        if run_in_databricks()
        else os.path.join(".", config_relative_path)
    )
    pls_trainer = PLSTrainer(config_path=config_path)
    pls_trainer.run()
