import discord
import asyncio
# from discord.ext import commands
from scipy.io import wavfile
import requests
import os
import torch
import sys
from dotenv import load_dotenv

from config.all_config import get_all_args

# llm part
from llm_utils.llm_utiils import Language_model
from sns.blue_sky.send_text import Sns_settings
from llm_utils.chatgpt_api import ChatGPTAPI
from llm_utils.claude_api import ClaudeAPI

# Style Bert Vits
from voice_utils.common.tts_model import ModelHolder
from voice_utils.infer import InvalidToneError
from voice_utils.text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize
from voice_utils.voice_utils import tts_fn

from voice_utils.common.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    GRADIO_THEME,
    LATEST_VERSION,
    Languages,
)


load_dotenv(verbose=True)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


# discordと接続した時に呼ばれる
@client.event
async def on_ready():
    client.args = get_all_args()

    print('Loading LLM model...')
    model_dir = os.path.join(client.args.lora_model_base_dir, client.args.model_instance_dir)
    llm_model = Language_model(client.args, client.args.llm_model_name, model_dir, client.args.tokenizer_name, "cuda")
    client.discord_llm_model = llm_model
    client.mafuyu_model = llm_model.prepare_models(quantization_type = "nf4",precision = torch.float16)
    client.mafuyu_tokenizer = llm_model.prepare_tokenizer()

    # load TTS model
    print('Loading TTS model...')
    # client.tts_model_dir = client.args.root_dir
    # client.model_names = client.args.voice_model_names
    client.model_holder = ModelHolder(client.args.root_dir, client.args.device) #root_dir: str, device: str
    if len(client.args.voice_model_names) == 0:
        print(
            f"Can not find models. Put models in {client.args.root_dir}."
        )
        sys.exit(1)

    client.model_holder.load_model(client.args.voice_model_names, os.path.join(client.args.root_dir, client.args.voice_model_names, client.args.safetensors_name)) #model_name: str, model_path: str 


    print(f'We have logged in as {client.user}')

# メッセージを受信した時に呼ばれる
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

    elif message.content.startswith('$ai'):
        words = message.content.split()
        input_prompt = ' '.join(words[1:])  # The message to be sent to the AI

        local_llm_prompt = client.discord_llm_model.prepare_prompt(input_prompt = input_prompt)
        final_prompt = f"{local_llm_prompt}"
        input_ids = client.mafuyu_tokenizer.encode(final_prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = client.mafuyu_model.generate(
            input_ids=input_ids.to(device="cuda"),
            max_length=500,
            temperature=0.7,
            do_sample=True,
        )
        output = client.mafuyu_tokenizer.decode(output_ids.tolist()[0][input_ids.size(1):])
        to_speach_text = client.discord_llm_model.refacter_prompt(output)

        ## voice inference part
        try:
            responce_message, (sr, audio), kata_tone_json_str =  tts_fn(
                        client.args.voice_model_names, #model_name
                        os.path.join(client.args.root_dir, client.args.voice_model_names, client.args.safetensors_name), #model_path
                        to_speach_text, #text_input
                        "JP", #language
                        None, #ref_audio_path
                        DEFAULT_SDP_RATIO, #sdp_ratio
                        DEFAULT_NOISE, #noise_scale
                        DEFAULT_NOISEW, #noise_scale_w
                        DEFAULT_LENGTH, #length_scale
                        DEFAULT_LINE_SPLIT, #line_split
                        DEFAULT_SPLIT_INTERVAL, #split_interval
                        None, #assist_text
                        DEFAULT_ASSIST_TEXT_WEIGHT, #assist_text_weight
                        False, #use_assist_text
                        DEFAULT_STYLE, #style
                        DEFAULT_STYLE_WEIGHT, #style_weight
                        None, #tone
                        False, #use_tone
                        client.args.speaker_name, #speaker
                        client.model_holder, #model_holder
                
                )
        except InvalidToneError or TypeError:
            to_speach_text = "ごめん...うまく聞こえなかったみたい"
            responce_message, (sr, audio), kata_tone_json_str =  tts_fn(
                        client.args.voice_model_names, #model_name
                        os.path.join(client.args.root_dir, client.args.voice_model_names, client.args.safetensors_name), #model_path
                        to_speach_text, #text_input
                        "JP", #language
                        None, #ref_audio_path
                        DEFAULT_SDP_RATIO, #sdp_ratio
                        DEFAULT_NOISE, #noise_scale
                        DEFAULT_NOISEW, #noise_scale_w
                        DEFAULT_LENGTH, #length_scale
                        DEFAULT_LINE_SPLIT, #line_split
                        DEFAULT_SPLIT_INTERVAL, #split_interval
                        None, #assist_text
                        DEFAULT_ASSIST_TEXT_WEIGHT, #assist_text_weight
                        False, #use_assist_text
                        DEFAULT_STYLE, #style
                        DEFAULT_STYLE_WEIGHT, #style_weight
                        None, #tone
                        False, #use_tone
                        client.args.speaker_name, #speaker
                        client.model_holder, #model_holder
                )

        
        await message.channel.send(to_speach_text)  # Send AI's response back to the Discord channel

        wavfile.write(client.args.save_audio_path, client.args.sampling_rate, audio)

        if os.path.exists(client.args.save_audio_path):
            await message.channel.send(file=discord.File(client.args.save_audio_path))
        else:
            await message.channel.send("Error: File does not exist.")
    
client.run(os.environ["DISCORD_TOKEN"])