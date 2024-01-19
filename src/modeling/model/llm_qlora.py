import mlflow
from typing import List, Dict, Union
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd


class LlmQlora(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained(
            context.artifacts["repository"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            context.artifacts["repository"],
            return_dict=True,
            load_in_4bit=True,
            device_map={"": 0},
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(
            base_model, context.artifacts["lora"]
        )

    @staticmethod
    def format_prompt(
        article: str, system_prompt: str, instruction: str
    ) -> str:
        instruction_prompt = f"""[INST]User: {instruction}
        #ARTICLE:
        {article}
        [/INST]"""
        full_prompt = "\n".join([system_prompt, instruction_prompt])
        return full_prompt

    def batch_input(
        self,
        articles: List[str],
        system_prompts: List[str],
        instructions: List[str],
        batch_size: int,
    ) -> List[str]:
        prompts = [
            LlmQlora.format_prompt(article, system_prompt, instruction)
            for article, system_prompt, instruction in zip(
                articles, system_prompts, instructions
            )
        ]
        return [
            prompts[idx : idx + batch_size]
            for idx in range(0, len(prompts), batch_size)
        ]

    def extract_output_texts(self, generated_texts: List[str]) -> List[str]:
        output_start_marker = "[/INST]"
        output_texts = []
        for output in generated_texts:
            start = output.index(output_start_marker) + len(
                output_start_marker
            )
            output_texts.append(output[start:])
        return output_texts

    def predict(
        self,
        context,
        model_input: Dict[str, Union[bool, int, List[str]]],
    ) -> List[str]:
        articles = model_input["articles"]
        system_prompts = model_input["system_prompts"]
        instructions = model_input["instructions"]
        temperature = model_input.get("temperature", 1.0)[0]
        max_tokens = model_input.get("max_tokens", 610)[0]
        top_p = model_input.get("top_p", 0.7)[0]
        num_return_sequences = model_input.get("num_return_sequences", 1)[0]
        do_sample = model_input.get("do_sample", True)[0]
        batch_size = model_input.get("batch_size", 1)[0]
        batched_prompts = self.batch_input(
            articles, system_prompts, instructions, batch_size
        )
        outputs = []
        for prompts in batched_prompts:
            model_inputs = self.tokenizer(
                prompts, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            with torch.cuda.amp.autocast():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            outputs.extend(self.extract_output_texts(generated_texts))
        return outputs


def predict(loaded_model, articles, params=None):
    system_prompt = (
        "You create laymen summaries of highly technical articles"
        " created by the biomedical industry. Your summary should"
        " cover most the following topics: Introduction or"
        " problem statement, experimental methods, "
        "results of experiments, discussions, recommended"
        " future work. The summary should have a max length"
        " of 600 tokens and should only use the "
        "information provided and no prior knowledge. "
        "Your output should be a summary with NO headers"
        " and NO bullet points, just a long text"
    )
    instruction = (
        "Create a plain language summary for this scientific article provided"
    )
    default_values = {
        "temperature": 1.0,
        "max_tokens": 610,
        "top_p": 0.7,
        "num_return_sequences": 1,
        "do_sample": True,
        "batch_size": 5,
        "system_prompts": system_prompt,
        "instructions": instruction,
    }
    if params:
        default_values.update(params)
    text_example = pd.DataFrame({"articles": articles})
    for key, value in default_values.items():
        text_example[key] = value
    text_example = text_example.astype(
        {"num_return_sequences": "int32", "batch_size": "int32"}
    )

    return loaded_model.predict(text_example)
