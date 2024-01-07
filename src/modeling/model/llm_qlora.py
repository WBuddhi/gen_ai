import mlflow
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    def predict(self, context, model_input):
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]
        batch = self.tokenizer(
            prompt, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda")
        with torch.cuda.amp.autocast():
            output_tokens = self.model.generate(
                input_ids=batch.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.7,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        )

        return generated_text
