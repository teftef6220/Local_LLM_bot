import argparse

def cfg():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model_name',type = str, default="rinna/japanese-gpt-neox-3.6b-instruction-sft", help='model')
    parser.add_argument('--model_base_dir',type = str, default="./models", help='model')
    parser.add_argument('--model_instance_dir',type = str, default="result_mafuyu", help='model')
    # parser.add_argument('--ckpt_name',type = str, default="model.ckpt", help='model')

    parser.add_argument('--tokenizer_path',type = str, default="rinna/japanese-gpt-neox-3.6b-instruction-sft", help='model')

    parser.add_argument('--prompt',type = str, default="貴方の名前を教えて？", help='model')

    parser.add_argument('--sns_type',type = str, default="blue_sky", help='sns_type')

    return parser.parse_args()