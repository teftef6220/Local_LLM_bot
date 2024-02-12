from config.all_config import get_all_args
from scipy.io import wavfile
import torch
import os
import sys
import yaml
import re

# whisper part
from whisper_wrapper import KeyControlledRecorder
import keyboard

# llm part
from llm_utiils import Language_model
from sns.blue_sky.send_text import Sns_settings

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

def main():
    # whisper part
    if args.use_whisper == True:
        print("use_whisper to Convert your voice")
        converter = KeyControlledRecorder(args.whisper_type)
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

    ## llm part
    
    model_dir = os.path.join(args.model_base_dir, args.model_instance_dir)
    
    llm_model = Language_model(args, args.llm_model_name, model_dir, args.tokenizer_name, "cuda")

    mafuyu_model = llm_model.prepare_models(quantization_type = "nf4",precision = torch.float16)

    mafuyu_tokenizer = llm_model.prepare_tokenizer()

    input_prompt = llm_model.prepare_prompt(prompt = input_prompt)

    final_prompt = f"""指示:\n{input_prompt}\n応答:"""

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

    to_speach_text = re.sub(r"</s>$", "", output)

    ##voice part

    model_dir = args.root_dir
    model_names = args.voice_model_names
    model_holder = ModelHolder(model_dir, args.device) #root_dir: str, device: str
    
    if len(model_names) == 0:
        print(
            f"Can not find models. Put models in {model_dir}."
        )
        sys.exit(1)

    #load model
    model_holder.load_model(model_names, os.path.join(model_dir, model_names, args.safetensors_name)) #model_name: str, model_path: str 

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
    
    sampling_rate = 44100
    output_file = "output.wav"
    wavfile.write(output_file, sampling_rate, audio)

    audio_buffer = io.BytesIO()
    write(audio_buffer, sampling_rate, audio)

    audio_buffer.seek(0)
    
    # simpleaudioを使ってバッファから直接再生
    wave_obj = sa.WaveObject.from_wave_file(audio_buffer)
    play_obj = wave_obj.play()
    play_obj.wait_done()



if __name__ == "__main__":
    args = get_all_args()
    main()