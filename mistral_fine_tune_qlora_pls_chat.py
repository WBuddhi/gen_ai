# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune mistralai/Mistral-7B-Instruct-v0.2 with QLORA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/huggingface/peft.git
# MAGIC %pip install bitsandbytes==0.41.3.post2 einops==0.6.1 trl==0.4.7
# MAGIC %pip install torch==2.0.1 accelerate==0.21.0 transformers==4.36.0
# MAGIC %pip install -U datasets
# MAGIC %pip install psutil==5.9.7 pynvml==11.5.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from mlflow import create_experiment, set_experiment, get_experiment_by_name, MlflowException
experiment_name = "/mistral_7b_instruct_v1_pls_gpt_summary_bz1_test"
try:
    experiment_id = create_experiment(experiment_name)
except MlflowException:
    experiment_id = get_experiment_by_name(
        experiment_name
    ).experiment_id
set_experiment(experiment_id=experiment_id)

# COMMAND ----------

experiment_id

# COMMAND ----------

from huggingface_hub import login

# Login to Huggingface to get access to the model
login(token="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", add_to_git_credential=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC
# MAGIC Using GPT detailed summary for PLS dataset in feature store at 

# COMMAND ----------

from datasets import Dataset
df = spark.table("llama2_feature_store.pls.gpt_summary")
dataset = Dataset.from_spark(df)

# COMMAND ----------

SYSTEM_PROMPT = "{system_prompt}".format(system_prompt="{system_prompt}")
INSTRUCTION_PROMPT = """[INST]User: Create a plain language summary for this scientific article
#ARTICLE:
{article}
[/INST]""".format(article="{article}")
def apply_prompt_template(examples):
    system_prompt = "You create laymen summaries of highly technical articles created by the biomedical industry"
    article = examples.get("gpt_summary")
    response = examples["summary"]
    full_prompt = "\n".join([SYSTEM_PROMPT.format(system_prompt=system_prompt), INSTRUCTION_PROMPT.format(article=article), response])
    return { "text": full_prompt }

# COMMAND ----------

dataset = dataset.map(apply_prompt_template)
print(dataset["text"][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the model
# MAGIC
# MAGIC In this section we will load the mistralai/Mistral-7B-Instruct-v0.2, quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.use_cache = False

# COMMAND ----------

lens = [len(i) for i in tokenizer(dataset["text"])["input_ids"]]

# COMMAND ----------

max(lens)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the configuration file in order to create the LoRA model. 
# MAGIC
# MAGIC According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. Therefore we will add `dense`, `dense_h_to_4_h` and `dense_4h_to_h` layers in the target modules in addition to the mixed query key value layer.

# COMMAND ----------

from peft import LoraConfig

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj'] # Choose all linear layers from the model
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the trainer

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# COMMAND ----------

from transformers import TrainingArguments
from mlflow import enable_system_metrics_logging

enable_system_metrics_logging()

torch.cuda.empty_cache()
output_dir = experiment_name
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 25
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
    report_to="mlflow",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Then finally pass everthing to the trainer

# COMMAND ----------

from trl import SFTTrainer

max_seq_length = 3200

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's train the model! Simply call `trainer.train()`

# COMMAND ----------

# trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the LORA model

# COMMAND ----------

# model_save_path = "/Volumes/llama2_feature_store/models/mistral_7b_instruct_v1_pls_gpt_summary_bz1_test"
# trainer.save_model(model_save_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the fine tuned model to MLFlow

# COMMAND ----------

# import torch
# from peft import PeftModel, PeftConfig

# peft_model_id =model_save_path
# config = PeftConfig.from_pretrained(peft_model_id)

# from huggingface_hub import snapshot_download
# # Download the Llama-2-7b-hf model snapshot from huggingface
# snapshot_location = snapshot_download(repo_id=config.base_model_name_or_path)


# COMMAND ----------

SYSTEM_PROMPT = "{system_prompt}".format(system_prompt="{system_prompt}")
INSTRUCTION_PROMPT = """[INST]User: Create a plain language summary for this scientific article
#ARTICLE:
{article}
[/INST]""".format(article="{article}")
def generate_prompt(examples):
    system_prompt = "You create laymen summaries of highly technical articles created by the biomedical industry"
    article = examples.get("gpt_summary")
    full_prompt = "\n".join([SYSTEM_PROMPT.format(system_prompt=system_prompt), INSTRUCTION_PROMPT.format(article=article)])
    return full_prompt

# COMMAND ----------

import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

class LLAMAQLORA(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['repository'])
    self.tokenizer.pad_token = self.tokenizer.eos_token
    config = PeftConfig.from_pretrained(context.artifacts['lora'])
    base_model = AutoModelForCausalLM.from_pretrained(
      context.artifacts['repository'], 
      return_dict=True, 
      load_in_4bit=True, 
      device_map={"":0},
      trust_remote_code=True,
    )
    self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])
  
  def predict(self, context, model_input):
    prompt = model_input["prompt"][0]
    temperature = model_input.get("temperature", [1.0])[0]
    max_tokens = model_input.get("max_tokens", [100])[0]
    batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
      output_tokens = self.model.generate(
          input_ids = batch.input_ids, 
          max_new_tokens=max_tokens,
          temperature=temperature,
          top_p=0.7,
          num_return_sequences=1,
          do_sample=True,
          pad_token_id=self.tokenizer.eos_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
      )
    generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
import mlflow
from huggingface_hub import snapshot_download

import torch
from peft import PeftModel, PeftConfig

with mlflow.start_run():
    trainer.train()

    model_save_path = "/Volumes/llama2_feature_store/models/mistral_7b_instruct_v1_pls_gpt_summary_bz1_test"
    trainer.save_model(model_save_path)

    peft_model_id =model_save_path
    config = PeftConfig.from_pretrained(peft_model_id)


    # Download the mistral-2-7b-hf model snapshot from huggingface
    snapshot_location = snapshot_download(repo_id=config.base_model_name_or_path)

    # Define input and output schema
    input_schema = Schema([
        ColSpec(DataType.string, "prompt"), 
        ColSpec(DataType.double, "temperature"), 
        ColSpec(DataType.long, "max_tokens")])
    output_schema = Schema([ColSpec(DataType.string)])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    prompt = generate_prompt(dataset[0])
    # Define input example
    temperature = [0.5]
    max_tokens = [1000]
    input_example=pd.DataFrame({
                "prompt": [prompt], 
                "temperature": temperature,
                "max_tokens": max_tokens})


    mlflow.pyfunc.log_model(
        "model",
        python_model=LLAMAQLORA(),
        artifacts={'repository' : snapshot_location, "lora": peft_model_id},
        # pip_requirements=["torch", "transformers", "accelerate", "einops", "loralib", "bitsandbytes", "peft"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.

# COMMAND ----------

import mlflow
import pandas as pd

# Load model as a PyFuncModel.
run_id = "0844d5bd02df4b6492953b20b1dca68c"
logged_model = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(logged_model)

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": temperature,
            "max_tokens": max_tokens})

# Predict on a Pandas DataFrame.
loaded_model.predict(text_example)

# COMMAND ----------


