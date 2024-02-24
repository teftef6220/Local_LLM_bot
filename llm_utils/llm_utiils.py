import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel


class Language_model():
    def __init__(self,args, model_name, model_dir_path, tokenizer_path, device):
        
        self.args = args    
        self.model_name = model_name
        self.model_dir_path = model_dir_path
        self.tokenizer_path = tokenizer_path
        self.device = device

    def prepare_models(self,quantization_type,precision):

        if quantization_type == "nf4":
            bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,  # 4bitベースモデルの有効化
                        bnb_4bit_quant_type=quantization_type,  # 量子化種別 (fp4 or nf4)
                        bnb_4bit_compute_dtype=precision,  # 4bitベースモデルのdtype (float16 or bfloat16)
                        bnb_4bit_use_double_quant=False,  # 4bitベースモデルのネストされた量子化の有効化 (二重量子化)
                    )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map={"": 0}
                    )
        elif quantization_type == "fp8":
            
            base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        load_in_8bit=True,
                        device_map={"": 0}
                    )
        
        model = PeftModel.from_pretrained(
                    base_model,
                    self.model_dir_path
                )
        
        
        return model
    
    def prepare_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    add_eos_token=True,
                    trust_remote_code=True
                )
        return tokenizer
    

    def prepare_prompt(self,input_prompt):
        """
        prepare prompt for LLM model(gemma model or other model)
        """
        if "gemma" in self.args.llm_model_name: # if gemma model
            final_prompt = f"""<bos><start_of_turn>user\n{input_prompt}<end_of_turn>\n<start_of_turn>model"""
        else:
            final_prompt = f"""指示:\n{input_prompt}\n応答:"""

        return final_prompt
    
    def refacter_prompt(self,input_text):
        """
        refacter prompt for output text

        # TODO:
            maniuplate output text for each model
            Do someones know better way to do this?

            ここはあとで治す...
            正規表現なんかいい方法ないですか？
        """
        if "gemma" in self.args.llm_model_name: # if gemma model
            final_text = re.sub(r"(<end_of_turn>\n<start_of_turn>model\n\n|<eos>)", "", input_text).strip()

        elif "swallow" in self.args.llm_model_name: # if swallow model
            pattern = r"\n(.*?)</s>"
            matches = re.findall(pattern, input_text, re.DOTALL)
            if matches:
                final_text = matches[-1].strip() 
            else:
                final_text = final_text

        elif "rinna" in self.args.llm_model_name: # if rinna model
            pattern = r"応答:(.*?)</s>"
            matches = re.findall(pattern, input_text, re.DOTALL)
            if matches:
                final_text = matches[-1].strip() 
            else:
                final_text = final_text

        else:
            final_text = input_text

        return final_text