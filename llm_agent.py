from config.all_config import get_all_args
from scipy.io import wavfile
import torch
import os
import sys
import yaml
import re

# Database
from DB.database import Database 
import datetime
from sqlalchemy.orm import sessionmaker, scoped_session

# whisper part
from whisper_utils.whisper_wrapper import KeyControlledRecorder
import keyboard

# llm part
from llm_utils.llm_utiils import Language_model
from sns.blue_sky.send_text import Sns_settings
from llm_utils.chatgpt_api import ChatGPTAPI
from llm_utils.claude_api import ClaudeAPI

# voice part
from atproto import Client, client_utils
from dotenv import load_dotenv
from scipy.io.wavfile import write
import simpleaudio as sa
import io

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

db = Database('sqlite:///DB/chat_sessions.db')

def prepare_memory(args,memory_num):
    memory_prompt = ""
    print("fetching data from DB")
    question_l,answer_l = db.get_memory(args.speaker_name,memory_num)
    for idx , answer_txt in enumerate(answer_l):
        memory_prompt += f"\n{question_l[idx]}\n{answer_txt}"
    return memory_prompt

def main():
    '''
    AI agent main function
    The process is as follows:
    whisper  →  llm  →  TTS

    input : args
        default : args.use_whisper = True
                    args.use_ChatGPT = False
    output : audio file

    you can set configs in config/all_config.py

    '''

    #load llm model if not use ChatGPT
    
    if args.use_ChatGPT: # if use ChatGPT
        print('Loading ChatGPT model...')
    elif args.use_claude_3: # if use Claude 3
        print('Loading Claude 3 model...')
    else: # if use Local LLM
        print('Loading LLM model...')
        model_dir = os.path.join(args.lora_model_base_dir, args.model_instance_dir)
        llm_model = Language_model(args, args.llm_model_name, model_dir, args.tokenizer_name, "cuda")
        mafuyu_model = llm_model.prepare_models(quantization_type = "nf4",precision = torch.float16)
        mafuyu_tokenizer = llm_model.prepare_tokenizer()
        

    #load TTS model
    print('Loading TTS model...')
    model_dir = args.root_dir
    model_names = args.voice_model_names
    model_holder = ModelHolder(model_dir, args.device) #root_dir: str, device: str
    if len(model_names) == 0:
        print(
            f"Can not find models. Put models in {model_dir}."
        )
        sys.exit(1)

    model_holder.load_model(model_names, os.path.join(model_dir, model_names, args.safetensors_name)) #model_name: str, model_path: str 


    ## whisper inference part
    if args.use_whisper == True : # if use whisper . when use whisper, you can use OpenAI whisper and see whisper_wapper.py
        print("use_whisper to Convert your voice")
        converter = KeyControlledRecorder(args.whisper_type,use_openai_whisper = args.use_OpenAI_whisper)
        keyboard.on_press(converter.on_press)
        keyboard.on_release(converter.on_release)
        print(f"Press '{converter.key}' to start and stop recording. Press 'esc' to exit.")
        keyboard.wait('esc')  # ESCキーで終了
        if converter.is_recording:
            converter.stop_recording()
        converter.p.terminate()
        input_prompt = converter.convert_to_text()
        os.remove(converter.output_filename)
    else:
        input_prompt = args.prompt


    if args.use_Memory: # if use Memory
        memory_prompt = prepare_memory(args,2)
    else:
        memory_prompt = ""

    ## llm inference part    
    if args.use_ChatGPT:
        final_prompt = f"{memory_prompt}\n{input_prompt}"
        chatgpt = ChatGPTAPI()
        output = chatgpt.chat(final_prompt,fine_tune=args.use_finetuning_GPT)
    elif args.use_claude_3:
        final_prompt = f"{memory_prompt}\n{input_prompt}"
        claude3 = ClaudeAPI()
        output = claude3.chat(final_prompt)
    else:
        local_llm_prompt = llm_model.prepare_prompt(input_prompt = input_prompt)
        final_prompt = f"{memory_prompt}\n{local_llm_prompt}"
        input_ids = mafuyu_tokenizer.encode(final_prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = mafuyu_model.generate(
            input_ids=input_ids.to(device=llm_model.device),
            max_length=500,
            temperature=0.7,
            do_sample=True,
        )
        output = mafuyu_tokenizer.decode(output_ids.tolist()[0][input_ids.size(1):])

    print(final_prompt)
    print("-------------------")
    print(output)

    if args.use_ChatGPT: # if use ChatGPT
        to_speach_text = output
    elif args.use_claude_3: # if use Claude 3
        to_speach_text = output
    else: # if use Local LLM
        to_speach_text = llm_model.refacter_prompt(output)

    ## voice inference part
    try:
        message, (sr, audio), kata_tone_json_str =  tts_fn(
                    model_names, #model_name
                    os.path.join(model_dir, model_names, args.safetensors_name), #model_path
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
                    args.speaker_name, #speaker
                    model_holder, #model_holder
            
            )
    except InvalidToneError:
        to_speach_text = "ごめん...うまく聞こえなかったみたい"
        message, (sr, audio), kata_tone_json_str =  tts_fn(
                    model_names, #model_name
                    os.path.join(model_dir, model_names, args.safetensors_name), #model_path
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
                    args.speaker_name, #speaker
                    model_holder, #model_holder
            )
    
    wavfile.write(args.save_audio_path, args.sampling_rate, audio)

    audio_buffer = io.BytesIO()
    write(audio_buffer, args.sampling_rate, audio)

    audio_buffer.seek(0)
    
    # simpleaudioを使ってバッファから直接再生
    wave_obj = sa.WaveObject.from_wave_file(audio_buffer)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    #save in database
    

    db.add_session(args.speaker_name,datetime.datetime.now(),args.use_whisper,args.llm_model_name,input_prompt,output)



if __name__ == "__main__":
    args = get_all_args()
    main()