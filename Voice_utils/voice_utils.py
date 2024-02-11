from infer import infer, get_net_g, infer_multilang
import utils
import numpy as np
from config import config
import re_matching
from tools.sentence import split_by_language
import datetime
import json
import warnings
from typing import Dict, List, Optional, Union
import argparse

# from common.log import logger
from common.tts_model import ModelHolder
from infer import InvalidToneError
from text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize

import torch
import sys
import os
import enum
import librosa
import yaml

from scipy.io import wavfile

from common.constants import (
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

# Get path settings
with open(os.path.join("./Voice_utils/configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]

languages = [l.value for l in Languages]

def tts_fn(
    model_name,
    model_path,
    text,
    language,
    reference_audio_path,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    line_split,
    split_interval,
    assist_text,
    assist_text_weight,
    use_assist_text,
    style,
    style_weight,
    kata_tone_json_str,
    use_tone,
    speaker,
):
    # model_holder.load_model(model_name, model_path)

    wrong_tone_message = ""
    kata_tone: Optional[list[tuple[str, int]]] = None
    if use_tone and kata_tone_json_str != "":
        if language != "JP":
            print("Only Japanese is supported for tone generation.")
            wrong_tone_message = "アクセント指定は現在日本語のみ対応しています。"
        if line_split:
            print("Tone generation is not supported for line split.")
            wrong_tone_message = (
                "アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"
            )
        try:
            kata_tone = []
            json_data = json.loads(kata_tone_json_str)
            # tupleを使うように変換
            for kana, tone in json_data:
                assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                kata_tone.append((kana, tone))
        except Exception as e:
            print(f"Error occurred when parsing kana_tone_json: {e}")
            wrong_tone_message = f"アクセント指定が不正です: {e}"
            kata_tone = None

    # toneは実際に音声合成に代入される際のみnot Noneになる
    tone: Optional[list[int]] = None
    if kata_tone is not None:
        phone_tone = kata_tone2phone_tone(kata_tone)
        tone = [t for _, t in phone_tone]

    speaker_id = model_holder.current_model.spk2id[speaker]

    start_time = datetime.datetime.now()

    try:
        sr, audio = model_holder.current_model.infer(
            text=text,
            language=language,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise_scale,
            noisew=noise_scale_w,
            length=length_scale,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=use_assist_text,
            style=style,
            style_weight=style_weight,
            given_tone=tone,
            sid=speaker_id,
        )
    except InvalidToneError as e:
        print(f"Tone error: {e}")
        return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str
    except ValueError as e:
        print(f"Value error: {e}")
        return f"Error: {e}", None, kata_tone_json_str

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    if tone is None and language == "JP":
        # アクセント指定に使えるようにアクセント情報を返す
        norm_text = text_normalize(text)
        kata_tone = g2kata_tone(norm_text)
        kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
    elif tone is None:
        kata_tone_json_str = ""
    message = f"Success, time: {duration} seconds."
    print(message)
    if wrong_tone_message != "":
        message = wrong_tone_message + "\n" + message
    return message, (sr, audio), kata_tone_json_str



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--root_dir", "-d", type=str, help="Model directory default is assetroot", default="./Voice_models"
    )
    parser.add_argument("--model_names", "-m", type=str, help="Model names to use", default="mafuyu")
    parser.add_argument("--device", type=str, help="Device to use", default="cuda")
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
    args = parser.parse_args()
    model_dir = args.root_dir

    model_holder = ModelHolder(args.root_dir, args.device) #root_dir: str, device: str

    model_names = args.model_names
    if len(model_names) == 0:
        print(
            f"モデルが見つかりませんでした。{model_dir}にモデルを置いてください。"
        )
        sys.exit(1)



    styles,speakers = model_holder.load_model(model_names, os.path.join(model_dir, model_names, "mafuyu_e100_s2000.safetensors")) #model_name: str, model_path: str 
    ##return styles[0] , speakers[0]をしているが後で


    message, (sr, audio), kata_tone_json_str =  tts_fn(
                "mafuyu", #model_name
                os.path.join(model_dir, model_names, "mafuyu_e100_s2000.safetensors"), #model_path
                "私の名前は....うん...朝比奈まふゆ", #text_input
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
                "mafuyu", #speaker

        )
    
    ##audio 処理

    sampling_rate = 44100

    # audio変数は、既にint16型のnumpy配列として定義されていると仮定しています。
    # 例: audio = np.array([413, 447, 468, ..., -145, -230, -246], dtype=np.int16)

    # WAVファイルとして書き出し
    output_file = "output.wav"
    wavfile.write(output_file, sampling_rate, audio)


