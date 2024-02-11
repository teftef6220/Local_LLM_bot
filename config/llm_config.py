import argparse

def cfg():

    #LLM Config
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model_name',type = str, default="rinna/japanese-gpt-neox-3.6b-instruction-sft", help='model')
    parser.add_argument('--model_base_dir',type = str, default="./llm_models", help='model')
    parser.add_argument('--model_instance_dir',type = str, default="result_mafuyu", help='model')
    # parser.add_argument('--ckpt_name',type = str, default="model.ckpt", help='model')

    parser.add_argument('--tokenizer_path',type = str, default="rinna/japanese-gpt-neox-3.6b-instruction-sft", help='model')

    parser.add_argument('--prompt',type = str, default="あなたの名前は？", help='model')

    parser.add_argument('--sns_type',type = str, default="blue_sky", help='sns_type')
    parser.add_argument('--LLM_type',type = str, default="Local", help='Local or GPT')


    # Voice Config
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=assets_root
    )
    parser.add_argument(
        "--share", action="store_true", help="Share this app publicly", default=False
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Server name for Gradio app",
    )
    parser.add_argument(
        "--no-autolaunch",
        action="store_true",
        default=False,
        help="Do not launch app automatically",
    )

    return parser.parse_args()