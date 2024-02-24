from config.all_config import get_all_args
from llm_utils.llm_utiils import Language_model
from sns.blue_sky.send_text import Sns_settings
import torch
import os

from atproto import Client, client_utils
from dotenv import load_dotenv



'''
llm に prompt を入力し, 生成されたテキストをsnsに送信する bot です
'''


def main():
    args = get_all_args()

    model_dir = os.path.join(args.lora_model_base_dir, args.model_instance_dir)
    
    llm_model = Language_model(args, args.llm_model_name, model_dir, args.tokenizer_name, "cuda")

    mafuyu_model = llm_model.prepare_models(quantization_type = "nf4",precision = torch.float16)

    mafuyu_tokenizer = llm_model.prepare_tokenizer()

    final_prompt = llm_model.prepare_prompt(input_prompt = args.prompt)

    # if "gemma" in args.llm_model_name:
    #     final_prompt = f"""<bos><start_of_turn>user\n{test_prompt}<end_of_turn>\n<start_of_turn>model"""
    # else:
    #     final_prompt = f"""指示:\n{test_prompt}\n応答:"""


    input_ids = mafuyu_tokenizer.encode(final_prompt, add_special_tokens=False, return_tensors="pt")

    output_ids = mafuyu_model.generate(
        input_ids=input_ids.to(device=llm_model.device),
        max_length=200,
        temperature=0.7,
        do_sample=True,
    )

    output = mafuyu_tokenizer.decode(output_ids.tolist()[0][input_ids.size(1):])

    print(final_prompt)
    print(output)

    print("send to SNS ...")
    # send_to_blusky(args,output)

def send_to_blusky(args,text):
    load_dotenv() ##to env
    if args.sns_type == "blue_sky":
        bluesky = Sns_settings(args.sns_type)
        bluesky = bluesky.login_to_blusky()

        bluesky.send_post(args.prompt+'\n'+str(text))



if __name__ == "__main__":
    main()