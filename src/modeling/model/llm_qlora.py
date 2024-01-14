from typing import List, Dict, Union
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


class LlmQlora:
    def __init__(self, model_path):
        lora = model_path
        config = PeftConfig.from_pretrained(model_path)
        repository = snapshot_download(repo_id=config.base_model_name_or_path)
        self.artifacts = {
            "repository": repository,
            "lora": lora,
        }
        self.load_context()

    def load_context(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifacts["repository"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            self.artifacts["repository"],
            return_dict=True,
            load_in_4bit=True,
            device_map={"": 0},
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(
            base_model, self.artifacts["lora"]
        )

    def format_prompt(self, article: str) -> str:
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
        instruction = "Create a plain language summary for this scientific article provided"
        instruction_prompt = f"""[INST]User: {instruction}
        #ARTICLE:
        {article}
        [/INST]"""
        full_prompt = "\n".join([system_prompt, instruction_prompt])
        return full_prompt

    def batch_input(self, articles: List[str], batch_size: int) -> List[str]:
        prompts = [self.format_prompt(article) for article in articles]
        return [
            prompts[idx : idx + batch_size]
            for idx in range(0, len(prompts), batch_size)
        ]

    def extract_output_texts(self, generated_texts: List[str]) -> List[str]:
        output_start_marker = "[/INST]"
        output_texts = []
        for output in generated_texts:
            print("cleaning text\n", output)
            start = output.index(output_start_marker) + len(
                output_start_marker
            )
            output_texts.append(output[start:])
        return output_texts

    def predict(
        self,
        model_input: Dict[str, Union[bool, int, List[str]]],
    ) -> List[str]:
        articles = model_input["articles"]
        temperature = model_input.get("temperature", 1.0)
        max_tokens = model_input.get("max_tokens", 610)
        top_p = model_input.get("top_p", 0.7)
        num_return_sequences = model_input.get("num_return_sequences", 1)
        do_sample = model_input.get("do_sample", True)
        batch_size = model_input.get("batch_size", 1)
        batched_prompts = self.batch_input(articles, batch_size)
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
