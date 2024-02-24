import argparse


def common_args(parser):
    
    parser.add_argument('--use_whisper', type=bool, default=False, help='use whisper or not')
    parser.add_argument('--use_ChatGPT', type=bool, default=False, help='use ChatGPT or not')
    parser.add_argument('--save_audio_path', type=str, default="output.wav", help='save audio path')
    parser.add_argument('--whisper_type', type=str, default='medium', help=['small', 'medium', 'large', 'tiny',"large"])
    parser.add_argument('--sampling_rate', type=int, default=44100, help='sampling rate')

def add_llm_args(parser):

    '''
    LoRAデータの配置
    llm_models
    |
    |---model_instance_dir
        |
        |---adapter_model.bin
        |---adapter_model.json

    ローカル dir から LLM を呼ぶときは huggingface path の代わりに dir path を記入してください。
    '''

    parser.add_argument('--llm_model_name', type=str, default="./llm_base_models/gemma-2b-it", help='base model name for example "rinna/japanese-gpt-neox-3.6b-instruction-sft"')
    parser.add_argument('--tokenizer_name', type=str, default="./llm_base_models/gemma-2b-it", help='base tokenizer name for example "rinna/japanese-gpt-neox-3.6b-instruction-sft"')
    parser.add_argument('--lora_model_base_dir', type=str, default="./llm_models", help='lora model base directory')
    parser.add_argument('--model_instance_dir', type=str, default="result_mafuyu_gemma_2b", help='model instance directory')
    parser.add_argument('--prompt', type=str, default="あなたについて教えて", help='prompt text')
    parser.add_argument('--sns_type', type=str, default="blue_sky", help='SNS type')
    parser.add_argument('--LLM_type', type=str, default="Local", help='LLM type')

def add_voice_args(parser):

    '''
    データの配置
    Voice_models
    |
    |---model_name
        |
        |---model_name_e100_s2000.safetensors
        |---config.json
        |---style_vectors.npy
    '''

    parser.add_argument("--root_dir", type=str,  default="./Voice_models", help="Model directory default is assetroot")
    parser.add_argument("--voice_model_names", type=str,default="mafuyu", help="Model names to use")
    parser.add_argument("--safetensors_name", type=str,default="mafuyu_e100_s2000.safetensors", help="SafeTensors model name")
    parser.add_argument("--speaker_name", type=str, default="mafuyu", help="Speaker name")

    parser.add_argument("--device", type=str,default="cuda", help="Device to use")
    parser.add_argument("--share", action="store_true", default=False, help="Share this app publicly")
    parser.add_argument("--no-autolaunch",action="store_true", default=False, help="Do not launch app automatically",)



def get_all_args():
    parser = argparse.ArgumentParser(description='共通の設定')
    common_args(parser)
    add_llm_args(parser)
    add_voice_args(parser)
    
    args = parser.parse_args()

    return args
