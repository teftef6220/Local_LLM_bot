import os
import torch
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
from trl import SFTTrainer


class Language_model():
    def __init__(self,args, model_name, model_dir_path, tokenizer_path, device):
        
        self.args = args    
        self.model_name = model_name
        self.model_dir_path = model_dir_path
        self.tokenizer_path = tokenizer_path
        self.device = device

    def prepare_models(self,quantization_type,precision):
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
    

    def prepare_prompt(self,prompt):
        prompt = prompt
        return prompt