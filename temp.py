# Built-in theme: "default", "base", "monochrome", "soft", "glass"
# See https://huggingface.co/spaces/gradio/theme-gallery for more themes
GRADIO_THEME: str = "NoCrypt/miku"

LATEST_VERSION: str = "2.2"
DEFAULT_STYLE: str = "Neutral"
DEFAULT_STYLE_WEIGHT: float = 5.0


class Languages(str, enum.Enum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


DEFAULT_SDP_RATIO: float = 0.2
DEFAULT_NOISE: float = 0.6
DEFAULT_NOISEW: float = 0.8
DEFAULT_LENGTH: float = 1.0
DEFAULT_LINE_SPLIT: bool = True
DEFAULT_SPLIT_INTERVAL: float = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT: float = 0.7
DEFAULT_ASSIST_TEXT_WEIGHT: float = 1.0

# hps = utils.get_hparams_from_file(config.webui_config.config_path)
hps = utils.get_hparams_from_file("./Voice_models/mafuyu_voice/config.json")#directory: 'Data\\model_name\\config.json'
# 若config.json中未指定版本则默认为最新版本
version = hps.version if hasattr(hps, "version") else LATEST_VERSION
# net_g = get_net_g(
#     model_path=config.webui_config.model, version=version, device="cuda", hps=hps
# )
net_g = get_net_g(
    model_path="./Voice_models/mafuyu_voice/models/G_2000.pth", version=version, device="cuda", hps=hps
)
speaker_ids = hps.data.spk2id
speakers = list(speaker_ids.keys())
languages = ["ZH", "JP", "EN", "mix", "auto"]


############## ここまでガロードの設定 ####################

def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = idx != 0
            skip_end = idx != len(slices) - 1
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device="cuda",
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
    return audio_list

def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
    reference_audio,
    emotion,
    style_text,
    style_weight,
):
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    text = text.replace("|", "")
    para_list = re_matching.cut_para(text)
    para_list = [p for p in para_list if p != ""]
    audio_list = []
    for p in para_list:
        if not cut_by_sent:
            audio_list += process_text(
                p,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
        else:
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            sent_list = [s for s in sent_list if s != ""]
            for s in sent_list:
                audio_list_sent += process_text(
                    s,
                    speaker,
                    sdp_ratio,
                    noise_scale,
                    noise_scale_w,
                    length_scale,
                    language,
                    reference_audio,
                    emotion,
                    style_text,
                    style_weight,
                )
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (hps.data.sampling_rate, audio_concat))


def process_mix(slice):
    _speaker = slice.pop()
    _text, _lang = [], []
    for lang, content in slice:
        content = content.split("|")
        content = [part for part in content if part != ""]
        if len(content) == 0:
            continue
        if len(_text) == 0:
            _text = [[part] for part in content]
            _lang = [[lang] for part in content]
        else:
            _text[-1].append(content[0])
            _lang[-1].append(lang)
            if len(content) > 1:
                _text += [[part] for part in content[1:]]
                _lang += [[lang] for part in content[1:]]
    return _text, _lang, _speaker


def process_auto(text):
    _text, _lang = [], []
    for slice in text.split("|"):
        if slice == "":
            continue
        temp_text, temp_lang = [], []
        sentences_list = split_by_language(slice, target_languages=["zh", "ja", "en"])
        for sentence, lang in sentences_list:
            if sentence == "":
                continue
            temp_text.append(sentence)
            if lang == "ja":
                lang = "jp"
            temp_lang.append(lang.upper())
        _text.append(temp_text)
        _lang.append(temp_lang)
    return _text, _lang


def process_text(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    style_text=None,
    style_weight=0,
):
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        for slice in re_matching.text_matching(text):
            _text, _lang, _speaker = process_mix(slice)
            if _speaker is None:
                continue
            print(f"Text: {_text}\nLang: {_lang}")
            audio_list.extend(
                generate_audio_multilang(
                    _text,
                    sdp_ratio,
                    noise_scale,
                    noise_scale_w,
                    length_scale,
                    _speaker,
                    _lang,
                    reference_audio,
                    emotion,
                )
            )
    elif language.lower() == "auto":
        _text, _lang = process_auto(text)
        print(f"Text: {_text}\nLang: {_lang}")
        audio_list.extend(
            generate_audio_multilang(
                _text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                _lang,
                reference_audio,
                emotion,
            )
        )
    else: ##今回は JP
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
        )
    return audio_list


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    prompt_mode,
    style_text=None,
    style_weight=0,
):
    if style_text == "":
        style_text = None
    if prompt_mode == "Audio prompt":
        if reference_audio == None:
            return ("Invalid audio prompt", None)
        else:
            reference_audio = load_audio(reference_audio)[1]
    else:
        reference_audio = None

    audio_list = process_text(
        text,
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language,
        reference_audio,
        emotion,
        style_text,
        style_weight,
    )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)


def format_utils(text, speaker):
    _text, _lang = process_auto(text)
    res = f"[{speaker}]"
    for lang_s, content_s in zip(_lang, _text):
        for lang, content in zip(lang_s, content_s):
            res += f"<{lang.lower()}>{content}"
        res += "|"
    return "mix", res[:-1]


def load_audio(path):
    audio, sr = librosa.load(path, 48000)
    # audio = librosa.resample(audio, 44100, 48000)
    return sr, audio