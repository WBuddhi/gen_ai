import os
from typing import Dict, Tuple, List

import torch
import pandas as pd
from datasets import Dataset
from mlflow import (
    MlflowException,
    create_experiment,
    enable_system_metrics_logging,
    log_table,
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
from src.modeling.model.llm_qlora import LlmQlora, predict
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
        self.__dict__.update(self.config["prompt"])
        self.experiment_id = self.mlflow_setup()
        torch.cuda.empty_cache()
        enable_system_metrics_logging()

    def _model_input_schema(self):
        return Schema(
            [
                ColSpec(DataType.string, "articles"),
                ColSpec(DataType.string, "system_prompts"),
                ColSpec(DataType.string, "instructions"),
                ColSpec(DataType.double, "temperature"),
                ColSpec(DataType.long, "max_tokens"),
                ColSpec(DataType.double, "top_p"),
                ColSpec(DataType.integer, "num_return_sequences"),
                ColSpec(DataType.boolean, "do_sample"),
                ColSpec(DataType.integer, "batch_size"),
            ]
        )

    def _model_output_schema(self):
        return Schema([ColSpec(DataType.string)])

    def _model_input_example(self):
        input_example = pd.DataFrame(
            {"article": [self.dataset["train"]["gpt_summary"][0]]}
        )
        input_example["system_prompts"] = self.system_prompt
        input_example["instructions"] = self.instruction
        input_example["temperature"] = 1.0
        input_example["max_tokens"] = 610
        input_example["top_p"] = 0.7
        input_example["num_return_sequences"] = 1
        input_example["do_sample"] = True
        input_example["batch_size"] = 10
        return input_example

    def _test_inference_model(self, run_id:str):
        articles = [self.dataset["test"]["gpt_summary"][0]]
        logged_model = f"runs:/{run_id}/{self.model_package['artifact_path']}"
        loaded_model = pyfunc.load_model(logged_model)
        return articles, predict(loaded_model=loaded_model, articles=articles, params={"batch_size": 5})

    def mlflow_setup(self):
        try:
            experiment_id = create_experiment(self.experiment_name)
        except MlflowException:
            logger.info("Experiment already exists")
            experiment_id = get_experiment_by_name(
                self.experiment_name
            ).experiment_id
        set_experiment(experiment_id=experiment_id)

    def format_prompt(self, example: Dict[str, str]) -> Dict[str, str]:
        article = example.get("gpt_summary")
        response = example.get("summary")
        full_prompt = LlmQlora.format_prompt(
            article, self.system_prompt, self.instruction
        )
        full_prompt = "\n".join(
            [
                full_prompt,
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
                    "gpt_summary", "summary"
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
        self.dataset = self.load_dataset()
        train_dataset = self.dataset["train"].map(self.format_prompt)
        model, tokenizer = self.model()
        peft_config = LoraConfig(**self.lora_config)
        training_args = TrainingArguments(
            **self.training_params,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            args=training_args,
            tokenizer=tokenizer,
            **self.sft_trainer_config,
        )
        return trainer

    def test_saved_model(self, run_id:str, output_file_path:str) -> List[str]:
        logger.info("Testing packaged model")
        torch.cuda.empty_cache()
        articles, predictions = self._test_inference_model(run_id)
        df = pd.DataFrame({"articles":articles, "predicted_summary": predictions})
        logger.info(f"Writing outputs to {output_file_path}")
        log_table(data=df, artifact_file=output_file_path)

    def run(self):
        with start_run() as run:
            trainer = self.trainer()
            trainer.train()
            logger.info("Training completed")

            run_id = run.info.run_id
            model_save_path = os.path.join(self.model_save_path, run_id)

            output_file_path = "output_file/output.json"
            trainer.save_model(model_save_path)
            logger.info(f"Saving model to: {model_save_path}")
            peft_model_id = model_save_path
            config = PeftConfig.from_pretrained(peft_model_id)

            snapshot_location = snapshot_download(
                repo_id=config.base_model_name_or_path,
            )

            signature = ModelSignature(
                inputs=self._model_input_schema(),
                outputs=self._model_output_schema(),
            )
            input_example = self._model_input_example()

            logger.info("Creating MLFlow model")
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
            return self.test_saved_model(run_id, output_file_path)



if __name__ == "__main__":
    config_relative_path = (
        "src/pipeline_configs/mistral_2_7b_instruct_pls.yaml"
    )
    config_path = (
        os.path.join(os.environ["REPO_ROOT_PATH"], config_relative_path)
        if run_in_databricks()
        else os.path.join(".", config_relative_path)
    )
    pls_trainer = PLSTrainer(config_path=config_path)
    pls_trainer.run()
