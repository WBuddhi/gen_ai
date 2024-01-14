import os
from typing import Dict, Tuple

import torch
from datasets import Dataset, IterableDataset
from mlflow import (autolog, create_experiment, get_experiment_by_name,
                    set_experiment, start_run)
from mlflow.exceptions import MlflowException
from peft import LoraConfig
from pyspark.sql.functions import col
from transformers import (BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer,
                          TrainingArguments)
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
            df = spark.table(f"{full_schema_name}.{table_name}").select(
                    col("model_input")
                ).sample(0.1)
            datasets[table_name] = Dataset.from_spark(df)
        return datasets

    def setup_hf_mlflow(self):
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = 1

    def model(self, modules:List[str]) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
        self.bits_and_bytes_config["bnb_4bit_compute_dtype"] = getattr(
            torch, self.bits_and_bytes_config["bnb_4bit_compute_dtype"]
        )
        quant_config = BitsAndBytesConfig(**self.bits_and_bytes_config)
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map={"": 0},
            use_flash_attention_2=self.use_flash_attention_2,
            cache_dir=self.hf_cache_dir,
            target_modules=modules,
        )
        model.config.use_cache = self.model_config["use_cache"]
        model.config.pretraining_tp = self.model_config["pretraining_tp"]

        tokenizer = LlamaTokenizer.from_pretrained(
            self.base_model, legacy=False, padding_side= self.padding_side, cache_dir=self.hf_cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    
    def find_all_linear_names(self, model):
        cls = bitsandbytes.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    
    def print_trainable_parameters(self, model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def trainer(self) -> SFTTrainer:
        model, tokenizer = self.model()
        modules = self.find_all_linear_names(model)
        peft_config = LoraConfig(**self.lora_config, target_modules=modules)
        training_args = TrainingArguments(**self.training_params, )
        self.print_trainable_parameters(model, use_4bit=True)
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
